import typing
import pufferlib.models
import math
import warnings

from typing import Optional, Sequence, Union

class Recurrent(pufferlib.models.RecurrentWrapper):
    def __init__(self, env, policy, input_size=512  , hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print
import numpy as np
DEVICE_TYPING = Union[torch.device, str, int]
if hasattr(typing, "get_args"):
    DEVICE_TYPING_ARGS = typing.get_args(DEVICE_TYPING)
else:
    DEVICE_TYPING_ARGS = (torch.device, str, int)
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter
#import utils
"""
BlurPool layer inspired by
 - Kornia's Max_BlurPool2d
 - Making Convolutional Networks Shift-Invariant Again :cite:`zhang2019shiftinvar`

Hacked together by Chris Ha and Ross Wightman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class BlurPool2d(nn.Module):
    r"""Creates a module that computes blurs and downsample a given feature map.
    See :cite:`zhang2019shiftinvar` for more details.
    Corresponds to the Downsample class, which does blurring and subsampling

    Args:
        channels = Number of input channels
        filt_size (int): binomial filter size for blurring. currently supports 3 (default) and 5.
        stride (int): downsampling filter stride

    Returns:
        torch.Tensor: the transformed tensor.
    """
    def __init__(self, channels, filt_size=3, stride=2) -> None:
        super(BlurPool2d, self).__init__()
        assert filt_size > 1
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride
        self.padding = [get_padding(filt_size, stride, dilation=1)] * 4
        coeffs = torch.tensor((np.poly1d((0.5, 0.5)) ** (self.filt_size - 1)).coeffs.astype(np.float32))
        blur_filter = (coeffs[:, None] * coeffs[None, :])[None, None, :, :].repeat(self.channels, 1, 1, 1)
        self.register_buffer('filt', blur_filter, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.padding, 'reflect')
        return F.conv2d(x, self.filt, stride=self.stride, groups=self.channels)
class CReLU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), dim=1)
    
    
'''
class VNetwork(nn.Module):
    def __init__(self, repr_dim, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(pufferlib.pytorch.layer_init(nn.Linear(repr_dim, feature_dim), std=1.0),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.V = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(feature_dim, hidden_dim) , std=1.0),
            nn.ReLU(inplace=True),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_dim, hidden_dim), std = 1.0),
            nn.ReLU(inplace=True),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )
                               
        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        v = self.V(h)
        return v
'''


class NoisyLinear(nn.Linear):
    """Noisy Linear Layer.

    Presented in "Noisy Networks for Exploration", https://arxiv.org/abs/1706.10295v3

    A Noisy Linear Layer is a linear layer with parametric noise added to the weights. This induced stochasticity can
    be used in RL networks for the agent's policy to aid efficient exploration. The parameters of the noise are learned
    with gradient descent along with any other remaining network weights. Factorized Gaussian
    noise is the type of noise usually employed.


    Args:
        in_features (int): input features dimension
        out_features (int): out features dimension
        bias (bool, optional): if ``True``, a bias term will be added to the matrix multiplication: Ax + b.
            Defaults to ``True``
        device (DEVICE_TYPING, optional): device of the layer.
            Defaults to ``"cpu"``
        dtype (torch.dtype, optional): dtype of the parameters.
            Defaults to ``None`` (default pytorch dtype)
        std_init (scalar, optional): initial value of the Gaussian standard deviation before optimization.
            Defaults to ``0.1``

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[DEVICE_TYPING] = None,
        dtype: Optional[torch.dtype] = None,
        std_init: float = 0.1,
    ):
        nn.Module.__init__(self)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = std_init

        self.weight_mu = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.weight_sigma = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.register_buffer(
            "weight_epsilon",
            torch.empty(out_features, in_features, device=device, dtype=dtype),
        )
        if bias:
            self.bias_mu = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.bias_sigma = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.register_buffer(
                "bias_epsilon",
                torch.empty(out_features, device=device, dtype=dtype),
            )
        else:
            self.bias_mu = None
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        if self.bias_mu is not None:
            self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: Union[int, torch.Size, Sequence]) -> torch.Tensor:
        if isinstance(size, int):
            size = (size,)
        x = torch.randn(*size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    @property
    def weight(self) -> torch.Tensor:
        if self.training:
            return self.weight_mu + self.weight_sigma * self.weight_epsilon
        else:
            return self.weight_mu

    @property
    def bias(self) -> Optional[torch.Tensor]:
        if self.bias_mu is not None:
            if self.training:
                return self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                return self.bias_mu
        else:
            return None


class NoisyLazyLinear(LazyModuleMixin, NoisyLinear):
    """Noisy Lazy Linear Layer.

    This class makes the Noisy Linear layer lazy, in that the in_feature argument does not need to be passed at
    initialization (but is inferred after the first call to the layer).

    For more context on noisy layers, see the NoisyLinear class.

    Args:
        out_features (int): out features dimension
        bias (bool, optional): if ``True``, a bias term will be added to the matrix multiplication: Ax + b.
            Defaults to ``True``.
        device (DEVICE_TYPING, optional): device of the layer.
            Defaults to ``"cpu"``.
        dtype (torch.dtype, optional): dtype of the parameters.
            Defaults to the default PyTorch dtype.
        std_init (scalar): initial value of the Gaussian standard deviation before optimization.
            Defaults to 0.1

    """

    def __init__(
        self,
        out_features: int,
        bias: bool = True,
        device: Optional[DEVICE_TYPING] = None,
        dtype: Optional[torch.dtype] = None,
        std_init: float = 0.1,
    ):
        super().__init__(0, 0, False, device=device)
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = UninitializedParameter(device=device, dtype=dtype)
        self.weight_sigma = UninitializedParameter(device=device, dtype=dtype)
        self.register_buffer(
            "weight_epsilon", UninitializedBuffer(device=device, dtype=dtype)
        )
        if bias:
            self.bias_mu = UninitializedParameter(device=device, dtype=dtype)
            self.bias_sigma = UninitializedParameter(device=device, dtype=dtype)
            self.register_buffer(
                "bias_epsilon", UninitializedBuffer(device=device, dtype=dtype)
            )
        else:
            self.bias_mu = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def reset_noise(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_noise()

    def initialize_parameters(self, input: torch.Tensor) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.in_features = input.shape[-1]
                self.weight_mu.materialize((self.out_features, self.in_features))
                self.weight_sigma.materialize((self.out_features, self.in_features))
                self.weight_epsilon.materialize((self.out_features, self.in_features))
                if self.bias_mu is not None:
                    self.bias_mu.materialize((self.out_features,))
                    self.bias_sigma.materialize((self.out_features,))
                    self.bias_epsilon.materialize((self.out_features,))
                self.reset_parameters()
                self.reset_noise()

    @property
    def weight(self) -> torch.Tensor:
        if not self.has_uninitialized_params() and self.in_features != 0:
            return super().weight

    @property
    def bias(self) -> torch.Tensor:
        if not self.has_uninitialized_params() and self.in_features != 0:
            return super().bias


def reset_noise(layer: nn.Module) -> None:
    """Resets the noise of noisy layers."""
    if hasattr(layer, "reset_noise"):
        layer.reset_noise()

class Policy(pufferlib.models.Policy):
    def __init__( self, env, 
                 frame_stack = 3, 
                 hidden_size = 512 , 
                 output_size = 512,
                 flat_size = 64*5*6,
                 downsample = 1 , 
                 adding_noise_lineay_layer = False, 
                 add_blur_pool: bool = False
                 ):
        super().__init__(env)
        
        self.flat_observation_space = env.flat_observation_space
        self.flat_observation_structure = env.flat_observation_structure
        print(
            "flat_observation_space: ", self.flat_observation_space,
            "\n"
            "flat_observation_structure: ", self.flat_observation_structure
        )
        
        self.observation_space = env.structured_observation_space
        self.screen_observation_space = self.observation_space["screen"]
        self.num_actions = env.action_space.n
        self.channels_last = True
        self.downsample = downsample
        self.nature_cnn = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(frame_stack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)) if not adding_noise_lineay_layer else pufferlib.pytorch.layer_init(NoisyLinear( in_features = 32*9*9 , out_features = 64)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)) if not adding_noise_lineay_layer else pufferlib.pytorch.layer_init(NoisyLinear( in_features = 64*7*7 , out_features = 64)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, hidden_size)) if not adding_noise_lineay_layer else pufferlib.pytorch.layer_init(NoisyLinear( in_features = flat_size , out_features = hidden_size)),
            nn.ReLU(),
        )
        self.pokemon_levels_embedding = nn.Sequential(
            nn.Embedding( num_embeddings = 128 , embedding_dim = 8),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear( in_features = 8*6 , out_features = 16)) if not adding_noise_lineay_layer else pufferlib.pytorch.layer_init(NoisyLinear( in_features = 8*6 , out_features = 16)),
            nn.ReLU(),
        )
        self.batle_status_embedding = nn.Sequential(
            nn.Embedding( num_embeddings = 4 , embedding_dim = 8),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear( in_features = 8 , out_features = 16)) if not adding_noise_lineay_layer else pufferlib.pytorch.layer_init(NoisyLinear( in_features = 8 , out_features = 16)),
            nn.ReLU(),
        )
        self.pokemon_and_oppoent_party_ids_embedding = nn.Sequential(
            nn.Embedding( num_embeddings = 256 , embedding_dim = 16),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear( in_features = 16*12 , out_features = 16)) if not adding_noise_lineay_layer else pufferlib.pytorch.layer_init(NoisyLinear( in_features = 16*12 , out_features = 16)),
            nn.ReLU(),
        )
        # poke_ids (12, ) -> (12, 8)
        #self.player_row_embedding = nn.Embedding(444,16)
        #self.player_column_embedding = nn.Embedding(436,16)
        self.last_projection = nn.LazyLinear( out_features = 512 ) if not adding_noise_lineay_layer else NoisyLazyLinear( out_features = 512 )
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(output_size, self.num_actions), std=0.01) # Policy 
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)
        #self.value_fn = VNetwork(output_size, 512, 512)
    def encode_observations(self, env_outputs):
        env_outputs = pufferlib.emulation.unpack_batched_obs(env_outputs,
            self.flat_observation_space, self.flat_observation_structure)
        if self.channels_last:
            observations = env_outputs["screen"].permute(0, 3, 1, 2)
            #print("observations: ", observations.shape)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        #return self.network(observations.float() / 255.0), None
        img = self.nature_cnn(observations.float() / 255.0)
        # Change the data tyupe of each pomemon lelve
        #print(env_outputs["each_pokemon_level"])
        each_pokemon_level = env_outputs["each_pokemon_level"]
        isinstance(each_pokemon_level, np.ndarray)
        pokemon_level_embedding = self.pokemon_levels_embedding(each_pokemon_level.to(torch.long))
        battle_embedding = self.batle_status_embedding(env_outputs["type_of_battle"].to(torch.long))
        concat = torch.cat((img, 
                                 env_outputs["party_size"] / 6.0 ,
                                 env_outputs["player_row"] / 444, 
                                 env_outputs["player_column"] / 436 , 
                                 env_outputs["party_health_ratio"],
                                 pokemon_level_embedding,
                                 battle_embedding,
                                 env_outputs["total_party_level"] / 600,
                                 ), dim=-1)
        assert not torch.isnan(concat).any()
        if concat.dtype != torch.float32:
            concat = concat.to(torch.float32)
        assert concat.dtype == torch.float32, f"concat.dtype: {concat.dtype} it should be torch.float32"
        
        return F.relu(self.last_projection(concat)), None
        

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
        