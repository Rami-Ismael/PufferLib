import pufferlib.models

class Recurrent(pufferlib.models.RecurrentWrapper):
    def __init__(self, env, policy, input_size=512  , hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print
import numpy as np
import utils

class CReLU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), dim=1)
    
    
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


class Policy(pufferlib.models.Policy):
    def __init__( self, env, 
                 frame_stack = 3, 
                 hiden_size = 512 , 
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
        
        self.nature_cnn = nn.Sequential(
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
        self.pokemon_levels_embedding = nn.Sequential(
            nn.Embedding( num_embeddings = 128 , embedding_dim = 8),
            nn.Flatten(),
            nn.Linear( in_features = 8*6 , out_features = 16),
            nn.ReLU(),
        )
        self.batle_status_embedding = nn.Sequential(
            nn.Embedding( num_embeddings = 4 , embedding_dim = 8),
            nn.Flatten(),
            nn.Linear( in_features = 8 , out_features = 16),
            nn.ReLU(),
        )
        self.pokemon_and_oppoent_party_ids_embedding = nn.Sequential(
            nn.Embedding( num_embeddings = 256 , embedding_dim = 16),
            nn.Flatten(),
            nn.Linear( in_features = 16*12 , out_features = 16),
            nn.ReLU(),
        )
        # poke_ids (12, ) -> (12, 8)
        #self.player_row_embedding = nn.Embedding(444,16)
        #self.player_column_embedding = nn.Embedding(436,16)
        self.last_projection = nn.LazyLinear( out_features = 512 )
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(output_size, self.num_actions), std=0.01)
        #self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)
        self.value_fn = VNetwork(output_size, 512, 512)
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
        