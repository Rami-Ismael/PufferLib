import pdb
from functools import partial
import torch
import torch
import torch.nn as nn
import pufferlib.models
import pufferlib.models
import pufferlib.pytorch

class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy,
            input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy,
            input_size, hidden_size, num_layers)
class ResnetBlock(torch.nn.Module):
    def __init__(self, in_planes, img_size=(15, 15)):
        super().__init__()
        self.model = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(torch.nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1)),
            torch.nn.LayerNorm((in_planes, *img_size)),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(torch.nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1)),
            torch.nn.LayerNorm((in_planes, *img_size)),
        )
    def forward(self, x):
        out = self.model(x)
        out += x
        return out

class Policy(nn.Module):
    def __init__(self, env, *args, framestack, flat_size,
            input_size=512, hidden_size=512, output_size=512,
            channels_last=False, downsample=1, **kwargs):
        '''The CleanRL default NatureCNN policy used for Atari.
        It's just a stack of three convolutions followed by a linear layer
        
        Takes framestack as a mandatory keyword argument. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__()
        self.channels_last = True
        self.downsample = downsample
        self.emulated = env.emulated
        print(f"The emulated environment is {self.emulated}")
        self.dtype = pufferlib.pytorch.nativize_dtype(self.emulated)
        print(f"The dtype is {self.dtype}")
        self.network = nn.Sequential(
            ResnetBlock( in_planes = 3 , img_size = (72, 80) ),
            nn.LeakyReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(17280, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
        )
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(output_size, 1), std=1)

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value , hidden

    def encode_observations(self, observations):
        env_outputs = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
        #pdb.set_trace()
        if self.channels_last:
            #observations = env_outputs["screen"].permute(0, 3, 1, 2)
            observations = env_outputs["screen"].permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        return self.network(observations.float() / 255.0), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value 