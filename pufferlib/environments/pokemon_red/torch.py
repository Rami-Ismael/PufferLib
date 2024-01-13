import pufferlib.models

class Recurrent:
    input_size:int = 512
    hidden_size:int = 512
    num_layers:int = 1

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print

class CReLU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), dim=1)
class Policy(pufferlib.models.Policy):
    def __init__( self, env, 
                 frame_stack = 3, 
                 hiden_size = 512, 
                 output_size = 512,
                 flat_size = 64*5*6,
                 downsample = 1
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
        
        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(frame_stack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, hiden_size)),
            nn.ReLU(),
        )
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(output_size, self.num_actions), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)
    def encode_observations(self, env_outputs):
        env_outputs = pufferlib.emulation.unpack_batched_obs(env_outputs,
            self.flat_observation_space, self.flat_observation_structure)
        if self.channels_last:
            observations = env_outputs["screen"].permute(0, 3, 1, 2)
            #print("observations: ", observations.shape)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        return self.network(observations.float() / 255.0), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
        