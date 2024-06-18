import math
from pdb import set_trace as T

import gymnasium
import functools

from pokegym import Environment

from gymnasium.wrappers.transform_reward import  TransformReward

import pufferlib.emulation
import pufferlib.postprocess


def env_creator(name='pokemon_red'):
    return functools.partial(make, name)

def make(name, headless: bool = True, state_path=None):
    '''Pokemon Red'''
    env = Environment(headless=headless, state_path=state_path)
    env = RenderWrapper(env)
    #env = pufferlib.postprocess.EpisodeStats(env) #we are removing this because I want to see the performances for each episode
    env = TransformReward(env, lambda x:  max(min(1.0 , x) , -1.0)) # clip the reward to -1 to 1 
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

class RenderWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        self.env = env

    @property
    def render_mode(self):
        return 'rgb_array'

    def render(self):
        return self.env.screen.screen_ndarray()
