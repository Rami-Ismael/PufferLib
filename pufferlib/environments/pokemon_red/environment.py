from pdb import set_trace as T

import gymnasium
import functools

from pokegym import Environment
from rich import print

import pufferlib.emulation


def env_creator(name='pokemon_red'):
    return functools.partial(make, name)


def make(name:str="pokemon_red", headless: bool = True, state_path=None , 
         punish_wipe_out:bool =True  ,
         framestack:int = 4
         ):
    '''Pokemon Red'''
    env = Environment(headless=headless, 
                      state_path=state_path , 
                        punish_wipe_out=punish_wipe_out
                      )
    if framestack != -1:
      env = gymnasium.wrappers.FrameStack(env, framestack)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env,
        postprocessor_cls=pufferlib.emulation.BasicPostprocessor)