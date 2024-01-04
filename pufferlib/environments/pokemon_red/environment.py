from pdb import set_trace as T

import gymnasium

from pokegym import Environment as env_creator

import pufferlib.emulation


def make_env(headless: bool = True, state_path=None , 
             reward_the_agent_for_completing_the_pokedex = False):
    '''Pokemon Red'''
    env = env_creator(headless=headless, state_path=state_path , 
                      reward_the_agent_for_the_number_of_pokemon_caught = reward_the_agent_for_completing_the_pokedex)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env,
        postprocessor_cls=pufferlib.emulation.BasicPostprocessor)
