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
    def __init__(self, env, *args,
            input_size=512, hidden_size=512, output_size=512,
            channels_last=False, downsample=1, 
            mlp_width = 512,
            mlp_depth = 3 , 
            **kwargs):
        '''The CleanRL default NatureCNN policy used for Atari.
        It's just a stack of three convolutions followed by a linear layer
        
        Takes framestack as a mandatory keyword argument. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__()
        self.channels_last = True
        self.downsample = downsample
        self.emulated = env.emulated
        for key, value in kwargs.items():
            print(f"{key}: {value}")
        print(f"The emulated environment is {self.emulated}")
        self.dtype = pufferlib.pytorch.nativize_dtype(self.emulated)
        print(f"The dtype is {self.dtype}")
        self.screen_network = nn.Sequential(
            ResnetBlock( in_planes = 3 , img_size = (72, 80) ),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(17280, hidden_size - 114 - 7 )),
            nn.LayerNorm(hidden_size - 114- 7 ),
            nn.ReLU(),
        )
        self.visited_and_global_mask = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d( in_channels = 1 ,  out_channels = 16, kernel_size = 8, stride = 4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d( in_channels = 16,  out_channels = 32, kernel_size = 4, stride = 2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d( in_channels = 32,  out_channels = 32, kernel_size = 3, stride = 1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear( 2560, hidden_size - 35)),
            nn.LayerNorm(hidden_size-35), 
            nn.ReLU(),
        )
        self.battle_stats_embedding = nn.Embedding(4 , 4, dtype=torch.float32)
        self.battle_results_embedding = nn.Embedding(4, 4, dtype=torch.float32)
        '''
            self.encode_linear = nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear( 1024 , mlp_width)),
                nn.LayerNorm(mlp_width),
                nn.ReLU(),
                pufferlib.pytorch.layer_init(nn.Linear( 1024 , hidden_size)),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
            )
        '''
        self.encode_linear = nn.Sequential()
        self.encode_linear.add_module(
                f"layer_0",
                pufferlib.pytorch.layer_init(nn.Linear( 1024 , mlp_width)),
            )
        self.encode_linear.add_module(
                f"layer_norm_0",
                nn.LayerNorm(mlp_width),
            )
        self.encode_linear.add_module(
                f"relu_0",
                nn.ReLU(),
            )
        for i in range( 1, mlp_depth-1):
            self.encode_linear.add_module(
                f"layer_{i}",
                pufferlib.pytorch.layer_init(nn.Linear( mlp_width , mlp_width)),
            )
            self.encode_linear.add_module(
                f"layer_norm_{i}",
                nn.LayerNorm(mlp_width),
            )
            self.encode_linear.add_module(
                f"relu_{i}",
                nn.ReLU(),
            )
        self.encode_linear.add_module(f"layer_{mlp_depth}",
            pufferlib.pytorch.layer_init(nn.Linear( mlp_width , hidden_size)))
        self.encode_linear.add_module(f"layer_norm_{mlp_depth}", nn.LayerNorm(hidden_size))
        self.encode_linear.add_module(f"relu_{mlp_depth}", nn.ReLU())
        
            
        
        
        self.selected_move_id =  nn.Embedding(
            166 , 
            8 , 
            dtype=torch.float32)
        self.map_music_sound_id_emebedding = nn.Embedding(76, 16, dtype=torch.float32)
        
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(output_size, 1), std=1)
        
                # pokemon has 0xF7 map ids
        # Lets start with 4 dims for now. Could try 8
        self.map_embeddings = torch.nn.Embedding(0xF7, 4, dtype=torch.float32)
        self.map_music_sound_bank_embeddings = torch.nn.Embedding(3, 6, dtype=torch.float32)
        self.pokemon_seen_fc = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(19, 16)),
            nn.LayerNorm(16),
            nn.ReLU(),
        )
        self.pokemon_caught_fc = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(19, 16)),
            nn.LayerNorm(16),
            nn.ReLU(),
        )
        self.pokemon_low_health_alarm = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(1, 2)),
            nn.LayerNorm(2),
            nn.ReLU(),
        )
        self.oppoents_pokemon_levels = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(6, 16)),
            nn.LayerNorm(16),
            nn.ReLU(),
        )
        self.total_pokemon_seen_fc = nn.Sequential(
            nn.Embedding(151 , 16, dtype=torch.float32),
            nn.LayerNorm(16) , 
            nn.ReLU()
        )
    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value , hidden
    def encode_observations(self, observations):
        env_outputs = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
        try:
            map_id = self.map_embeddings(env_outputs["map_id"].long())
        except Exception as e:
            print(e)
            pdb.set_trace()
        #pdb.set_trace()
        if self.channels_last:
            #observations = env_outputs["screen"].permute(0, 3, 1, 2)
            observations = env_outputs["screen"].permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        try:
            return self.encode_linear(
                torch.cat(
                    (
                    (self.screen_network(observations.float() / 255.0).squeeze(1)) ,
                    (self.visited_and_global_mask( torch.cat( (env_outputs["visited_mask"].permute(0, 3, 1, 2).float() , env_outputs["global_map"].permute(0, 3, 1, 2).float() ) , dim = -1) ).squeeze(1)),
                    env_outputs["x"].float() // 444,
                    env_outputs["y"].float() // 436,
                    map_id.squeeze(1),
                    self.map_music_sound_bank_embeddings(env_outputs["map_music_sound_bank"].long()).squeeze(1) , 
                    env_outputs["party_size"].float() / 6.0,
                    env_outputs["each_pokemon_level"].float() / 100.0,
                    env_outputs["total_party_level"].float() / 600.0  , 
                    env_outputs["number_of_turns_in_current_battle"].float() / 255.0 , 
                    env_outputs["total_party_level"].float() / env_outputs["party_size"].float() , # average level of party
                    env_outputs["each_pokemon_health_points"].float() , # average health of party
                    env_outputs["each_pokemon_max_health_points"].float() / 703.0 ,  # https://github.com/xinpw8/pokegym/blob/a8b75e4ad2694461f661acf5894d498b69d1a3fa/pokegym/bin/ram_reader/red_ram_api.py#L752
                    self.battle_stats_embedding(env_outputs["battle_stats"].long()).squeeze(1),
                    self.battle_results_embedding(env_outputs["battle_result"].long()).squeeze(1),
                    env_outputs["total_number_of_items"].float() / 64.0,
                    env_outputs["money"].float() / 999999.0,
                    self.selected_move_id(env_outputs["player_selected_move_id"].long()).squeeze(1),
                    self.selected_move_id(env_outputs["enemy_selected_move_id"].long()).squeeze(1) , 
                    self.map_music_sound_id_emebedding(env_outputs["map_music_sound_id"].long()).squeeze(1) , 
                    env_outputs["player_xp"].float() , 
                    env_outputs["total_party_max_hit_points"].float() , 
                    self.total_pokemon_seen_fc(env_outputs["total_pokemon_seen"].long()).squeeze(1) ,
                    self.pokemon_seen_fc(env_outputs["pokemon_seen_in_the_pokedex"].float() / 255.0).squeeze(1) , 
                    self.pokemon_caught_fc(env_outputs["byte_representation_of_caught_pokemon_in_the_pokedex"].float() / 255.0).squeeze(1) ,
                    self.pokemon_low_health_alarm(env_outputs["low_health_alarm"].float() / 255.0).squeeze(1) ,
                    self.oppoents_pokemon_levels(env_outputs["opponent_pokemon_levels"].float()/ 100.0).squeeze(1) ,
                    env_outputs["enemy_trainer_pokemon_hp"].float() / 705.0,
                    env_outputs["enemy_pokemon_hp"].float() / 705.0,
                    ) ,
                    dim = -1
                )
            ) , None
        except Exception as e:
            print(e)
            # Checks for each keys
            print(f"The Data types of env_outputs is {env_outputs.dtype}")
            print(f"Determine the amout of data that exist in GPU {torch.cuda.memory_allocated()}")
            pdb.set_trace()

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value 