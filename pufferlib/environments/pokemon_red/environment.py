import math
from pdb import set_trace as T

import gymnasium
import functools

from pokegym import Environment

from gymnasium.wrappers.transform_reward import  TransformReward

import pufferlib.emulation
import pufferlib.postprocess

from rich import print as print


def env_creator(name='pokemon_red'):
    return functools.partial(make, name)

def make(name, headless: bool = True, state_path=None , 
         display_info_interval_divisor = 2048 , 
         EpisodeStats = False , 
         max_episode_steps = 1024 , 
         max_reward_clip = 1.0, 
        reward_for_increase_pokemon_level_coef =  1.1 , 
        reward_for_increasing_the_highest_pokemon_level_in_the_team_by_battle_coef = 1,
        reward_for_explore_unique_coor_coef = .4 , 
        reward_for_entering_a_trainer_battle_coef = 1.0 ,
        propability_of_full_game_reset_at_reset = 0.1 ,
        random_starter_pokemon:bool = True  , 
        negative_reward_for_wiping_out_coef = 1.0,
        negative_reward_for_entering_a_trainer_battle_lower_total_pokemon_level_coef:float = 1.0 ,
        reward_for_using_bad_moves_coef:float =1.0 ,
        reward_for_increasing_the_total_party_level:float = 1.0 ,
        reward_for_knocking_out_wild_pokemon_by_battle_coef:float = 1.0 ,
        reward_for_doing_new_events:float = 1.0 ,
        level_up_reward_threshold:int = 4 , 
        multiple_exp_gain_by_n:int = 3 , 
        reward_for_finding_higher_level_wild_pokemon_coef:float = 1.0 , 
        reward_for_knocking_out_enemy_pokemon_in_trainer_party_coef:float = 1.0 , 
        reward_for_finding_new_maps_coef:float = 1.0 ,
        disable_wild_encounters:bool =  True,
        set_enemy_pokemon_damage_calcuation_to_zero = True,
         ):
    #print(f"The current diosplayin of interval divisor is {display_info_interval_divisor}")
    '''Pokemon Red'''
    env = Environment(headless=headless, 
                        state_path=state_path , 
                        display_info_interval_divisor=display_info_interval_divisor , 
                        max_episode_steps=max_episode_steps ,
                        reward_for_explore_unique_coor_coef=reward_for_explore_unique_coor_coef ,
                        reward_for_increase_pokemon_level_coef=reward_for_increase_pokemon_level_coef , 
                        random_starter_pokemon = random_starter_pokemon  , 
                        reward_for_entering_a_trainer_battle_coef = reward_for_entering_a_trainer_battle_coef , 
                        negative_reward_for_wiping_out_coef = negative_reward_for_wiping_out_coef,
                        negative_reward_for_entering_a_trainer_battle_lower_total_pokemon_level_coef = negative_reward_for_entering_a_trainer_battle_lower_total_pokemon_level_coef ,
                        reward_for_using_bad_moves_coef = reward_for_using_bad_moves_coef ,
                        disable_wild_encounters = disable_wild_encounters ,
                        level_up_reward_threshold = level_up_reward_threshold ,
                        reward_for_knocking_out_wild_pokemon_by_battle_coef = reward_for_knocking_out_wild_pokemon_by_battle_coef ,
                        reward_for_doing_new_events = reward_for_doing_new_events ,
                        reward_for_finding_new_maps_coef = reward_for_finding_new_maps_coef ,
                        multiple_exp_gain_by_n = multiple_exp_gain_by_n , 
                        reward_for_knocking_out_enemy_pokemon_in_trainer_party_coef = reward_for_knocking_out_enemy_pokemon_in_trainer_party_coef ,
                        reward_for_finding_higher_level_wild_pokemon_coef = reward_for_finding_higher_level_wild_pokemon_coef,
                        reward_for_increasing_the_total_party_level = reward_for_increasing_the_total_party_level ,
                        set_enemy_pokemon_damage_calcuation_to_zero  = set_enemy_pokemon_damage_calcuation_to_zero,
                        )
    env = RenderWrapper(env)
    if EpisodeStats:
        env = pufferlib.postprocess.EpisodeStats(env) #we are removing this because I want to see the performances for each episode
    env = TransformReward(env, lambda x:  max(min(max_reward_clip , x) , -1 * max_reward_clip)) # clip the reward to -1 to 1 
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

class RenderWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        self.env = env

    @property
    def render_mode(self):
        return 'rgb_array'

    def render(self):
        return self.env.screen.screen_ndarray()
