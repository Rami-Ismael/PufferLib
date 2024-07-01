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
         EpisodeStats = False , max_episode_steps = 65536 , 
         max_reward_clip = 1.0, 
        reward_for_increase_pokemon_level_coef =  1.1 , 
        reward_for_explore_unique_coor_coef = .4 , 
        random_starter_pokemon:bool = True
         ):
    #print(f"The current diosplayin of interval divisor is {display_info_interval_divisor}")
    '''Pokemon Red'''
    env = Environment(headless=headless, 
                      state_path=state_path , display_info_interval_divisor=display_info_interval_divisor , 
                      max_episode_steps=max_episode_steps , 
                      reward_for_explore_unique_coor_coef=reward_for_explore_unique_coor_coef ,
                        reward_for_increase_pokemon_level_coef=reward_for_increase_pokemon_level_coef , 
                        random_starter_pokemon = random_starter_pokemon )
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
