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
def create_screen_network(embedd_the_x_and_y_coordinate = False , hidden_size=512):
    outputs_of_the_screen_nework = 512
    if embedd_the_x_and_y_coordinate:
        outputs_of_the_screen_nework  -= 94 # 96 which is 48 *2 = 96 - 2 because we are placing a two elemnts in the process in the process 
    outputs_of_the_screen_nework -= 114 # idk
    outputs_of_the_screen_nework -=7 # idk
    return nn.Sequential(
            ResnetBlock( in_planes = 3 , img_size = (72, 80) ),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(17280, outputs_of_the_screen_nework )),
            nn.LayerNorm( outputs_of_the_screen_nework ),
            nn.ReLU(),
        )
def crete_mlp(dense_act_func: str = "ReLU", mlp_width:int = 512, mlp_depth:int = 3, hidden_size:int = 512):
        encode_linear = nn.Sequential()
        for i in range( 1, mlp_depth+2 ):
            if i == 0:
                encode_linear.add_module(
                    f"layer_{i}",
                    pufferlib.pytorch.layer_init(nn.Linear( 1024 , mlp_width)),
                )
            elif i == 1:
                encode_linear.add_module(
                    f"layer_{i}",
                    pufferlib.pytorch.layer_init(nn.Linear( mlp_width , mlp_width)),
                )
            elif i == mlp_depth+1:
                encode_linear.add_module(
                    f"layer_{i}",
                    pufferlib.pytorch.layer_init(nn.Linear( mlp_width , hidden_size)),
                )
            encode_linear.add_module(
                f"layer_norm_{i}",
                nn.LayerNorm(mlp_width),
            )
            if dense_act_func == "ReLU":
                encode_linear.add_module(
                    f"relu_{i}",
                    nn.ReLU(),
                )
            elif dense_act_func == "LeakyReLU":
                encode_linear.add_module(
                    f"leaky_relu_{i}",
                    nn.LeakyReLU(),
                )
        print(f"The encode linear layer is {self.encode_linear}")
        return encode_linear

class Policy(nn.Module):
    def __init__(self, env, *args,
            input_size=512, hidden_size=512, output_size=512,
            channels_last=False, downsample=1, 
            mlp_width = 512,
            mlp_depth = 3 , 
            embedd_the_x_and_y_coordinate = True,
            dense_act_func: str = "ReLU",
            **kwargs):
        '''The CleanRL default NatureCNN policy used for Atari.
        It's just a stack of three convolutions followed by a linear layer
        
        Takes framestack as a mandatory keyword argument. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__()
        self.channels_last = True
        self.downsample = downsample
        self.emulated = env.emulated
        self.embedd_the_x_and_y_coordinate = embedd_the_x_and_y_coordinate
        for key, value in kwargs.items():
            print(f"{key}: {value}")
        print(f"The emulated environment is {self.emulated}")
        self.dtype = pufferlib.pytorch.nativize_dtype(self.emulated)
        print(f"The dtype is {self.dtype}")
        self.screen_network = create_screen_network( embedd_the_x_and_y_coordinate = self.embedd_the_x_and_y_coordinate, hidden_size=hidden_size)
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
        self.encode_linear: nn.Sequential = crete_mlp(dense_act_func=dense_act_func, mlp_width=mlp_width, mlp_depth=mlp_depth, hidden_size=hidden_size)
            
        
        
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
        # https://forums.fast.ai/t/size-of-embedding-for-categorical-variables/42608/4
        self.coordinate_fc_x = nn.Sequential(
            nn.Embedding(444 , 48 , dtype=torch.float32),
            nn.LayerNorm(48),
            nn.ReLU()
        )
        self.coordinate_fc_y = nn.Sequential(
            nn.Embedding(436 , 48 , dtype=torch.float32),
            nn.LayerNorm(48),
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
        def test_embedding(add_embedding:bool = False):
            if add_embedding:
                return self.coordinate_fc_x(env_outputs["x"]) ,  self.coordinate_fc_y(env_outputs["y"])
            return env_outputs["x"].float() // 444 , env_outputs["y"].float() // 436
        try:
            elements_to_concatenate = [
                    self.screen_network(observations.float() / 255.0).squeeze(1) ,
                    self.visited_and_global_mask( torch.cat( (env_outputs["visited_mask"].permute(0, 3, 1, 2).float() , env_outputs["global_map"].permute(0, 3, 1, 2).float() ) , dim = -1) ).squeeze(1),
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
                    ]
            if self.embedd_the_x_and_y_coordinate:
                elements_to_concatenate.append(self.coordinate_fc_x(env_outputs["x"].int()).squeeze(1))
                elements_to_concatenate.append( self.coordinate_fc_y(env_outputs["y"].int()).squeeze(1))
            else:
                elements_to_concatenate.append(env_outputs["x"].float() // 444)
                elements_to_concatenate.append( env_outputs["y"].float() // 436)
            return self.encode_linear(torch.cat(elements_to_concatenate, dim = -1)) , None
        except Exception as e:
            print(e)
            # Checks for each keys
            print(f"The type of env_outputs is {type(env_outputs)}")
            pdb.set_trace()

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value 