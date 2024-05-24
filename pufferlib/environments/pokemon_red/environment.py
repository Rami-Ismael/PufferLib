from pdb import set_trace as T

import gymnasium
import functools

from pokegym import Environment
from rich import print

import pufferlib.emulation
import pufferlib.postprocess


def env_creator(name='pokemon_red'):
    return functools.partial(make, name)


def make(name:str="pokemon_red", headless: bool = True, state_path=None , 
         punish_wipe_out:bool =True  ,
         framestack:int = -1,
         ):
    '''Pokemon Red'''
    env = Environment(headless=headless, state_path=state_path)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)
