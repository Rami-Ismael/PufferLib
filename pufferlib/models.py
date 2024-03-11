from pdb import set_trace as T
from typing import Optional, Tuple
import numpy as np

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.distributions import Categorical

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces

import math


class Policy(nn.Module):
    '''Pure PyTorch base policy
    
    This spec allows PufferLib to repackage your policy
    for compatibility with RL frameworks

    encode_observations -> decode_actions is PufferLib's equivalent of PyTorch forward
    This structure provides additional flexibility for us to include an LSTM
    between the encoder and decoder.

    To port a policy to PufferLib, simply put everything from forward() before the
    recurrent cell (or, if no recurrent cell, everything before the action head)
    into encode_observations and put everything after into decode_actions.

    You can delete the recurrent cell from forward(). PufferLib handles this for you
    with its framework-specific wrappers. Since each frameworks treats temporal data a bit
    differently, this approach lets you write a single PyTorch network for multiple frameworks.

    Specify the value function in critic(). This is a single value for each batch element.
    It is called on the output of the recurrent cell (or, if no recurrent cell,
    the output of encode_observations)
    '''
    def __init__(self, env):
        super().__init__()
        if isinstance(env, pufferlib.emulation.GymnasiumPufferEnv):
            self.observation_space = env.observation_space
            self.action_space = env.action_space
        else:
            self.observation_space = env.single_observation_space
            self.action_space = env.single_action_space

        # Used to unflatten observation in forward pass
        self.unflatten_context = env.unflatten_context

        self.is_multidiscrete = isinstance(self.action_space,
                pufferlib.spaces.MultiDiscrete)

        if not self.is_multidiscrete:
            assert isinstance(self.action_space, pufferlib.spaces.Discrete)

    @abstractmethod
    def encode_observations(self, flat_observations):
        '''Encodes a batch of observations into hidden states

        Call pufferlib.emulation.unpack_batched_obs at the start of this
        function to unflatten observations to their original structured form:

        observations = pufferlib.emulation.unpack_batched_obs(
            env_outputs, self.unflatten_context)
 
        Args:
            flat_observations: A tensor of shape (batch, ..., obs_size)

        Returns:
            hidden: Tensor of (batch, ..., hidden_size)
            lookup: Tensor of (batch, ...) that can be used to return additional embeddings
        '''
        raise NotImplementedError

    @abstractmethod
    def decode_actions(self, flat_hidden, lookup):
        '''Decodes a batch of hidden states into multidiscrete actions

        Args:
            flat_hidden: Tensor of (batch, ..., hidden_size)
            lookup: Tensor of (batch, ...), if returned by encode_observations

        Returns:
            actions: Tensor of (batch, ..., action_size)
            value: Tensor of (batch, ...)

        actions is a concatenated tensor of logits for each action space dimension.
        It should be of shape (batch, ..., sum(action_space.nvec))'''
        raise NotImplementedError

    def forward(self, env_outputs):
        '''Forward pass for PufferLib compatibility'''
        hidden, lookup = self.encode_observations(env_outputs)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value
class FFA(nn.Module):
    def __init__(
        self,
        # Required
        memory_size: int,
        context_size: int,
        # Model Settings
        max_len: int = 1024,
        dtype: torch.dtype = torch.double,
        oscillate: bool = True,
        learn_oscillate: bool = True,
        decay: bool = True,
        learn_decay: bool = True,
        fudge_factor: float = 0.01,
        # Weight Init Settings
        min_period: int = 1,
        max_period: int = 1024,
        forgotten_at: float = 0.01,
        modify_real: bool = True,
    ):
        """A phaser-encoded aggregation operator
        Inputs:
            Required Settings:
                memory_size: Feature dimension of the model

            Model Settings:
                max_len: Maximum length of the batch in timesteps. Note this
                    is not the episode length, but rather the sequence length. The model
                    will be fastest if all sequences are within max_len. But we may
                    experience floating point under/overflow for very long sequences.
                    Setting this to less than the sequence length will break the
                    sequence into parts, trading off speed for accuracy. Gradients
                    will propagate as usual across the boundaries.
                context_size: The number of context filters for each channel
                    of memory_size.
                dtype: Whether to use floats or doubles. Note doubles enables
                    significantly more representational power for a little
                    extra compute.
                oscillate: Whether we should use the imaginary component of
                    the exponential (sinusoidal oscillations). If this is false,
                    the model cannot determine relative time between inputs.
                learn_oscillate: Whether the oscillate terms (omega in the paper)
                    should be learned.
                decay: Whether we should use the real component of the exponential.
                    If this is false, the model cannot decay inputs over time.
                learn_decay: Whether the decay terms (alpha in the paper)
                    should be learned.
                fudge_factor: A small positive number to prevent floating point
                    overflows.

            Weight Initialization Settings:
                forgetten_at: What fraction of the original input a memory is considered
                    "forgotten" at.
                min_period: The initial minimum sinusoidal period and trace decay. 
                    This is the minimum relative time distance the model can 
                    initially represent. Note that if min/max period are learned,
                    they can exceed the limits set here.
                max_period: The initial maximum sinusoidal period and trace decay. This 
                    determines the maximum relative time distance the model can initially
                    represent.
                modify_real: If this is false, min_period, max_period, and forgotten_at
                    will not affect the alpha term initialization.
        """
        super().__init__()
        self.memory_size = memory_size
        self.max_len = max_len
        self.context_size = context_size
        self.oscillate = oscillate
        self.learn_oscillate = learn_oscillate
        self.decay = decay
        self.learn_decay = learn_decay
        assert dtype in [torch.float, torch.double]
        self.dtype = dtype
        dtype_max = torch.finfo(dtype).max
        # To prevent overflows, ensure exp(limit * max_len) < {float,double}
        # limit * max_len < log({float,double})
        # limit == log({float,double}) / max_len - fudge_factor
        self.limit = math.log(dtype_max) / max_len - fudge_factor

        # Memories will be a fraction (epsilon) of their original value
        # at max_period
        # exp(a * max_period) < epsilon
        # a = < log(epsilon) / max_period
        if modify_real:
            soft_high = math.log(forgotten_at) / max_period
        else:
            soft_high = -1e-6

        # Initialize parameters
        real_param_shape = [1, 1, self.memory_size]
        imag_param_shape = [1, 1, self.context_size]
        a_low = -self.limit + fudge_factor
        a_high = max(min(-1e-6, soft_high), a_low)

        a = torch.linspace(a_low, a_high, real_param_shape[-1]).reshape(
            real_param_shape
        )
        b = (
            2
            * torch.pi
            / torch.linspace(min_period, max_period, imag_param_shape[-1]).reshape(
                imag_param_shape
            )
        )

        if not self.oscillate:
            b.fill_(1 / 1e6)
        if not self.decay:
            a.fill_(0)

        self.a = nn.Parameter(a)
        self.b = nn.Parameter(b)

        # For typechecker
        self.one: torch.Tensor
        self.inner_idx: torch.Tensor
        self.outer_idx: torch.Tensor
        self.state_offset: torch.Tensor
        # Buffers
        self.register_buffer("one", torch.tensor([1.0], dtype=torch.float))
        # n, n - 1, ..., 1, 0
        self.register_buffer("inner_idx", torch.arange(max_len, dtype=dtype).flip(0))
        # 0, -1, ..., -n + 1, -n
        self.register_buffer("outer_idx", -self.inner_idx)
        # 1, 2, ... n + 1
        self.register_buffer("state_offset", torch.arange(1, max_len + 1, dtype=dtype))

    def extra_repr(self):
        return f"in_features={self.memory_size}, out_features=({self.memory_size}, {self.context_size})"

    def log_gamma(self, t_minus_i: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        assert t_minus_i.dim() == 1
        T = t_minus_i.shape[0]
        if clamp:
            self.a.data = self.a.data.clamp(min=-self.limit, max=-1e-8)
            a = self.a.clamp(min=-self.limit, max=-1e-8)
        else:
            a = self.a
        b = self.b
        if not self.oscillate or not self.learn_oscillate:
            b = b.detach()
        if not self.decay or not self.learn_decay:
            a = a.detach()

        exp = torch.complex(
            a.reshape(1, 1, -1, 1),
            b.reshape(1, 1, 1, -1),
        )
        out = exp * t_minus_i.reshape(1, T, 1, 1)
        return out

    def gamma(self, t_minus_i: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        """Gamma function from the paper

        Args:
            t_minus_i: 1D tensor of relative time indices (can be discrete or cont.)
            log: Whether to return the gamma or log of gamma

        Returns:
            gamma^t for t in t_minus_i
        """
        return self.log_gamma(t_minus_i, clamp=clamp).exp()

    def compute_z(self, x):
        # TODO: See https://math.stackexchange.com/questions/1844525/logarithm-of-a-sum-or-addition
        B, T, F, D = x.shape
        return torch.cumsum(self.gamma(self.inner_idx[:T]) * x, dim=1)

    def batched_recurrent_update(
        self, x: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """Given x_{k:n} and s_{k-1}, compute s{k:n}

        Args:
            x: input of [B, T, memory_size]
            memory: state of [B, 1, memory_size, context_size]

        Returns
            state of [B, n-k, memory_size, context_size]
        """
        B, T, F, D = x.shape
        # inner_idx: n, n - 1, ..., 1, 0
        # outer_idx: -n, -n + 1, ... -1, 0
        # state_offset: 1, 2, ... n + 1

        # (gamma^{n}, gamma^{n-1}, ... gamma^{0}) \odot (x0, x1, ... xn)
        #z = torch.cumsum(self.gamma(self.inner_idx[:T]) * x, dim=1)
        z = self.compute_z(x)
        memory = self.gamma(self.outer_idx[:T]) * z + memory * self.gamma(
            self.state_offset[:T]
        )

        return memory.to(torch.complex64)

    def single_step_update(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """A fast recurrent update for a single timestep"""
        return x + memory * self.gamma(self.one, clamp=False)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            x: [B, T, F, 1]
            memory: [B, 1, F, dtype=torch.complex]
        Returns:
            memory: [B, 1, F, dtype=torch.complex]
        """
        assert memory.dtype in [
            torch.complex64,
            torch.complex128,
        ], "State should be complex dtype"
        assert x.dim() == 3
        assert memory.dim() == 4

        B, T, F = x.shape
        x = x.reshape(B, T, F, 1)

        if x.shape[1] == 1:
            # More efficient shortcut for single-timestep inference
            return self.single_step_update(x, memory)
        elif x.shape[1] < self.max_len:
            # Default case, the whole thing can fit into a single temporal batch
            return self.batched_recurrent_update(x, memory)
        else:
            # Need to break into temporal batches
            chunks = x.split(self.max_len, dim=1)
            states = []
            for chunk in chunks:
                memory = self.batched_recurrent_update(chunk, memory[:, -1:])
                states.append(memory)
            return torch.cat(states, dim=1)


class LogspaceFFA(FFA):
    """FFA but designed for very long sequences using logspace arithmetic"""
    # TODO: gamma is limited but need not be here
    def set_nonzero(self, x, eps=1e-10):
        """Set values (of memory) to be nonzero to prevent inf when taking the log"""
        x.real[x.real == 0] = eps
        mask = x.real.abs() < eps
        x.real[mask] = x.real[mask].sign() * eps
        return x

    def compute_z(self, x):
        # eq 4. https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        # https://math.stackexchange.com/questions/1538477/log-of-summation-expression
        B, T, F, D = x.shape
        log_divisor = self.log_gamma(self.inner_idx[T-1:T])
        log_z = log_divisor + torch.log(
            torch.cumsum(
                (x + self.log_gamma(self.inner_idx[:T]) - log_divisor).exp(),
            dim=1),
        )   
        return log_z

    def batched_recurrent_update(
        self, x: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """Given x_{k:n} and s_{k-1}, compute s{k:n}

        Args:
            x: input of [B, T, memory_size]
            memory: state of [B, 1, memory_size, context_size]

        Returns
            state of [B, n-k, memory_size, context_size]
        """
        B, T, F, D = x.shape
        # inner_idx: n, n - 1, ..., 1, 0
        # outer_idx: -n, -n + 1, ... -1, 0
        # state_offset: 1, 2, ... n + 1

        # (gamma^{n}, gamma^{n-1}, ... gamma^{0}) \odot (x0, x1, ... xn)
        log_z = self.compute_z(x)
        memory = torch.exp(self.log_gamma(self.outer_idx[:T], clamp=False) + log_z) + torch.exp(self.set_nonzero(memory).log() + self.log_gamma(
            self.state_offset[:T], clamp=False
        ))

        return memory.to(torch.complex64)

    def single_step_update(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """A fast recurrent update for a single timestep"""
        return x.exp() + torch.exp(self.set_nonzero(memory).log() + self.log_gamma(self.one, clamp=False))
class RecurrentWrapper(Policy):
    def __init__(self, env, policy, input_size=128, hidden_size=128, num_layers=1):
        super().__init__(env)

        if not isinstance(policy, Policy):
            raise ValueError('Subclass pufferlib.Policy to use RecurrentWrapper')

        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.recurrent = torch.nn.LSTM(
            input_size, hidden_size, num_layers=num_layers)

        for name, param in self.recurrent.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

    def forward(self, x, state):
        x_shape, space_shape = x.shape, self.observation_space.shape
        x_n, space_n = len(x_shape), len(space_shape)
        assert x_shape[-space_n:] == space_shape

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError('Invalid input tensor shape', x.shape)

        if state is not None:
            assert state[0].shape[1] == state[1].shape[1] == B

        x = x.reshape(B*TT, *space_shape)
        hidden, lookup = self.policy.encode_observations(x)
        assert hidden.shape == (B*TT, self.input_size), f"{hidden.shape} != {(B*TT, self.input_size)}"

        hidden = hidden.reshape(B, TT, self.input_size)
        hidden = hidden.transpose(0, 1)
        hidden, state = self.recurrent(hidden, state)

        hidden = hidden.transpose(0, 1)
        hidden = hidden.reshape(B*TT, self.hidden_size)

        hidden, critic = self.policy.decode_actions(hidden, lookup)
        return hidden, critic, state
class FFM(nn.Module):
    """Fast and Forgetful Memory

    Args:
        input_size: Size of input features to the model
        hidden_size: Size of hidden layers within the model
        memory_size: Size of the decay dimension of memory (m in the paper)
        context_size: Size of the temporal context (c in the paper, the
            total recurrent size is m * c)
        output_size: Output feature size of the model
        min_period: Minimum period for FFA, see FFA for details
        max_period: Maximum period for FFA, see FFA for details

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int,
        context_size: int,
        output_size: int,
        min_period: int = 1,
        max_period: int = 1024,
        logspace: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.context_size = context_size

        self.pre = nn.Linear(input_size, 2 * memory_size + 2 * output_size)
        if logspace:
            self.ffa = LogspaceFFA(
                memory_size=memory_size,
                context_size=context_size,
                min_period=min_period,
                max_period=max_period,
            )
        else:
            self.ffa = FFA(
                memory_size=memory_size,
                context_size=context_size,
                min_period=min_period,
                max_period=max_period,
            )
        self.mix = nn.Linear(2 * memory_size * context_size, output_size)
        self.ln_out = nn.LayerNorm(output_size, elementwise_affine=False)

    def initial_state(self, batch_size=1, device='cpu'):
        return torch.zeros(
            (batch_size, 1, self.memory_size, self.context_size),
            device=device,
            dtype=torch.complex64,
        )


    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input of shape [B, T, input_size]
            state: Recurrent state of size [B, 1, memory_size, context_size]
                and dtype=complex

        Returns:
            y: Output of shape [B, T, output_size]
            state: Recurrent state of size [B, 1, memory_size, context_size]
                and dtype=complex
        """

        B, T, _ = x.shape
        if state is None:
            # Typechecker doesn't like us to reuse the 'state' name
            s = self.initial_state(B, x.device)
        else:
            s = state

        # Compute values from x
        y, thru, gate = self.pre(x).split(
            [self.memory_size, self.output_size, self.memory_size + self.output_size],
            dim=-1,
        )

        # Compute gates
        gate = gate.sigmoid()
        in_gate, out_gate = gate.split([self.memory_size, self.output_size], dim=-1)

        # Compute state and output
        s = self.ffa((y * in_gate), s)  # Last dim for context
        z = self.mix(torch.view_as_real(s).reshape(B, T, -1))
        out = self.ln_out(z) * out_gate + thru * (1 - out_gate)

        return out, s
class DropInFFM(FFM):
    """Fast and Forgetful Memory, wrapped to behave like an nn.GRU

    Args:
        input_size: Size of input features to the model
        hidden_size: Size of hidden layers within the model
        memory_size: Size of the decay dimension of memory (m in the paper)
        context_size: Size of the temporal context (c in the paper, the
            total recurrent size is m * c)
        output_size: Output feature size of the model
        min_period: Minimum period for FFA, see FFA for details
        max_period: Maximum period for FFA, see FFA for details
        batch_first: Whether inputs/outputs/states are shape
            [batch, time, *]. If false, the inputs/outputs/states are
            shape [time, batch, *]

    """
    def __init__(self, *args, batch_first=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_first = batch_first
        self.num_layers = 1

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input of shape [B, T, F] or [B, F] if batch first,
                otherwise [T, B, F] or [T, F]
            state: Recurrent state of size [B, 1, M, C] or [B, M, C]
                and dtype=complex if batch_first, else
                [1, B, M, C] or [B, M, C]

        Returns:
            y: Output with the same batch dimensions as the input
            state: Recurrent state of the same shape as the input recurrent state
        """
        # Check if x missing singleton time or batch dim
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_y = True
        else:
            squeeze_y = False

        # Check if s is missing singleton time dim
        if self.batch_first:
            B, T, F = x.shape
        else:
            T, B, F = x.shape

        if state is None:
            # Typechecker doesn't like us to reuse the 'state' name
            s = self.initial_state(B, x.device)
        else:
            s = state

        s = s.reshape(B, 1, self.memory_size, self.context_size)

        # Sanity check shapes
        assert s.shape == (
            B,
            1,
            self.memory_size,
            self.context_size,
        ), (
            f"Given x of shape {x.shape}, expected state to be"
            f" shape [{B}, 1, {self.memory_size}, {self.context_size}], dtype=complex, but got {s.shape} "
            f"and dtype={s.dtype}"
        )   


        # Make everything batch first for FFM
        if not self.batch_first:
            x = x.permute(1, 0, 2)

        # Call FFM
        y, s = super().forward(x, s)

        if not self.batch_first:
            y = y.permute(1, 0, 2)

        if squeeze_y:
            y = y.reshape(B, -1)

        # Return only terminal state of s
        s = s[:, -1]

        return y, s

class DROP_IN_FM_WRAPPER(Policy):
    def __init__(self, env, policy, input_size=128, hidden_size=128, num_layers=1):
        super().__init__(env)

        if not isinstance(policy, Policy):
            raise ValueError('Subclass pufferlib.Policy to use RecurrentWrapper')

        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.recurrent = DropInFFM(
            input_size=input_size,
            hidden_size=hidden_size,
            memory_size=32,
            context_size=4,
            output_size=hidden_size,
            batch_first=True,
        )

        #for name, param in self.recurrent.named_parameters():
        #    if "bias" in name:
        #        nn.init.constant_(param, 0)
        #    elif "weight" in name:
        #        nn.init.orthogonal_(param, 1.0)

    def forward(self, x, state):
        x_shape, space_shape = x.shape, self.observation_space.shape
        x_n, space_n = len(x_shape), len(space_shape)
        assert x_shape[-space_n:] == space_shape

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError('Invalid input tensor shape', x.shape)

        if state is not None:
            assert state[0].shape[1] == state[1].shape[1] == B

        x = x.reshape(B*TT, *space_shape)
        hidden, lookup = self.policy.encode_observations(x)
        assert hidden.shape == (B*TT, self.input_size), f"{hidden.shape} != {(B*TT, self.input_size)}"

        hidden = hidden.reshape(B, TT, self.input_size)
        hidden = hidden.transpose(0, 1)
        hidden, state = self.recurrent(hidden, state)

        hidden = hidden.transpose(0, 1)
        hidden = hidden.reshape(B*TT, self.hidden_size)

        hidden, critic = self.policy.decode_actions(hidden, lookup)
        return hidden, critic, state
class Default(Policy):
    def __init__(self, env, input_size=128, hidden_size=128):
        '''Default PyTorch policy, meant for debugging.
        This should run with any environment but is unlikely to learn anything.
        
        Uses a single linear layer + relu to encode observations and a list of
        linear layers to decode actions. The value function is a single linear layer.
        '''
        super().__init__(env)
        self.encoder = nn.Linear(np.prod(self.observation_space.shape), hidden_size)

        if self.is_multidiscrete:
            self.decoders = nn.ModuleList([nn.Linear(hidden_size, n)
                for n in self.action_space.nvec])
        else:
            self.decoder = nn.Linear(hidden_size, self.action_space.n)

        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, env_outputs):
        '''Forward pass for PufferLib compatibility'''
        hidden, lookup = self.encode_observations(env_outputs)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        '''Linear encoder function'''
        hidden = observations.reshape(observations.shape[0], -1).float()
        hidden = torch.relu(self.encoder(hidden))
        return hidden, None

    def decode_actions(self, hidden, lookup, concat=True):
        '''Concatenated linear decoder function'''
        value = self.value_head(hidden)

        if self.is_multidiscrete:
            actions = [dec(hidden) for dec in self.decoders]
            return actions, value

        action = self.decoder(hidden)
        return action, value

class Convolutional(Policy):
    def __init__(self, env, *args, framestack, flat_size,
            input_size=512, hidden_size=512, output_size=512,
            channels_last=False, downsample=1, **kwargs):
        '''The CleanRL default Atari policy: a stack of three convolutions followed by a linear layer
        
        Takes framestack as a mandatory keyword arguments. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__(env)
        self.num_actions = self.action_space.n
        self.channels_last = channels_last
        self.downsample = downsample

        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(framestack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, hidden_size)),
            nn.ReLU(),
        )

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(output_size, self.num_actions), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)

    def encode_observations(self, observations):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        return self.network(observations.float() / 255.0), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

# ResNet Procgen baseline 
# https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

class ProcgenResnet(Policy):
    def __init__(self, env, cnn_width=16, mlp_width=256):
        super().__init__(env)
        h, w, c = env.structured_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [cnn_width, 2*cnn_width, 2*cnn_width]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=mlp_width),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(mlp_width, self.action_space.n), std=0.01)
        self.value = pufferlib.pytorch.layer_init(
                nn.Linear(mlp_width, 1), std=1)

    def encode_observations(self, x):
        x = pufferlib.emulation.unpack_batched_obs(x, self.unflatten_context)
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)
        return hidden, None
 
    def decode_actions(self, hidden, lookup):
        '''linear decoder function'''
        action = self.actor(hidden)
        value = self.value(hidden)
        return action, value
