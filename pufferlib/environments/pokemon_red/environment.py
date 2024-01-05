from pdb import set_trace as T

import gymnasium

from pokegym import Environment as env_creator

import pufferlib.emulation


def make_env(headless: bool = True, state_path=None , 
             reward_the_agent_for_completing_the_pokedex = True , 
             reward_the_agent_for_the_normalize_gain_of_new_money = True,
             ):
    '''Pokemon Red'''
    env = env_creator(headless=headless, state_path=state_path , 
                      reward_the_agent_for_completing_the_pokedex=  reward_the_agent_for_completing_the_pokedex, 
                        reward_the_agent_for_the_normalize_gain_of_new_money = reward_the_agent_for_the_normalize_gain_of_new_money)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env,
        postprocessor_cls=pufferlib.emulation.BasicPostprocessor)
