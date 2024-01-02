from argparse import Namespace
import argparse
import importlib
from types import ModuleType
from typing import Any
from rich import print
import sys
import pufferlib
import pufferlib.args
import pufferlib.utils
import pufferlib.models
from clean_pufferl import CleanPuffeRL, rollout


from demo import make_config


def test_pokegym_map():
    parser = argparse.ArgumentParser(description='Parse environment argument', add_help=False)
    parser.add_argument('--env', type=str, default='pokemon_red', help='Environment name')
    parser.add_argument('--vectorization', type=str, default='serial', help='Vectorization method (serial, multiprocessing, ray)')
    clean_parser = argparse.ArgumentParser(parents=[parser])
    args = parser.parse_known_args()[0].__dict__
    env = args['env']
    env_module , sweep_config , cfg = make_config(env)
    print(f'env_module: {env_module}')
    print(f'sweep_config: {sweep_config}')
    print(f'cfg: {cfg}')
    
    args: dict[str, dict[str, Any]] = {}
    
    for name, sub_config in cfg.items():
        args[name] = {}
        for key, value in sub_config.items():
            data_key = f'{name}.{key}'
            cli_key = f'--{data_key}'.replace('_', '-')
            if isinstance(value, bool) and value is False:
                action = 'store_false'
                parser.add_argument(cli_key, default=value, action='store_true')
                clean_parser.add_argument(cli_key, default=value, action='store_true')
            elif isinstance(value, bool) and value is True:
                data_key = f'{name}.no_{key}'
                cli_key = f'--{data_key}'.replace('_', '-')
                parser.add_argument(cli_key, default=value, action='store_false')
                clean_parser.add_argument(cli_key, default=value, action='store_false')
            else:
                parser.add_argument(cli_key, default=value, type=type(value))
                clean_parser.add_argument(cli_key, default=value, metavar='', type=type(value))

            args[name][key] = getattr(parser.parse_known_args()[0], data_key)
    clean_parser.parse_args(sys.argv[1:])
    args['args'] = pufferlib.args.CleanPuffeRL(**args['args'])
    args = pufferlib.namespace(**args)
    args.vectorization = pufferlib.vectorization.Serial
    args.seed = 1
    args.total_timesteps = 1
    args.torch_deterministic = True
    
    trainer = CleanPuffeRL(
        args,
        env_module,
        
    )
    
    print(f'trainer: {trainer}')
    

    
    





if __name__ == '__main__':
    test_pokegym_map()