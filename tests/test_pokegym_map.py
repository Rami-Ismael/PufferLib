from argparse import Namespace
import argparse
import importlib
from types import ModuleType
from typing import Any
from rich import print
import pufferlib
import inspect
import sys


class Serial:
    '''Runs environments in serial on the main process
    
    Use this vectorization module for debugging environments
    '''
    __init__ = serial_vec_env.init
    single_observation_space = property(vec_env.single_observation_space)
    single_action_space = property(vec_env.single_action_space)
    structured_observation_space = property(vec_env.structured_observation_space)
    flat_observation_space = property(vec_env.flat_observation_space)
    unpack_batched_obs = vec_env.unpack_batched_obs
    send = serial_vec_env.send
    recv = serial_vec_env.recv
    async_reset = serial_vec_env.async_reset
    profile = serial_vec_env.profile
    reset = serial_vec_env.reset
    step = serial_vec_env.step
    put = serial_vec_env.put
    get = serial_vec_env.get
    close = serial_vec_env.close

@pufferlib.dataclass
class SweepMetric:
    goal = 'maximize'
    name = 'episodic_return'
    
    
def make_sweep_config(method='random', name='sweep',
        metric=None, cleanrl=None, env=None, policy=None):
    sweep_parameters = {}
    if metric is None:
        sweep_metric = dict(SweepMetric())
    else:
        sweep_metric = dict(metric)

    if cleanrl is not None:
        sweep_parameters['cleanrl'] = {'parameters': dict(cleanrl)}
    if env is not None:
        sweep_parameters['env'] = {'parameters': dict(env)}
    if policy is not None:
        sweep_parameters['policy'] = {'parameters': dict(policy)}
        
    return {
        'method': method,
        'name': name,
        'metric': sweep_metric,
        'parameters': sweep_parameters,
    }
import torch
@pufferlib.dataclass
class CleanPuffeRL:
    seed: int = 1
    torch_deterministic: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_timesteps: int = 10_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    num_envs: int = 8
    envs_per_worker: int = 1
    envs_per_batch: int = None
    synchroneous: bool = False
    verbose: bool = True
    data_dir: str = 'experiments'
    checkpoint_interval: int = 200
    cpu_offload: bool = True
    pool_kernel: list = [0]
    batch_size: int = 1024
    batch_rows: int = 32
    bptt_horizon: int = 16 #8
    vf_clip_coef: float = 0.1 
def pokegym():
    import pufferlib
    args = CleanPuffeRL(
        total_timesteps=100_000_000,
        num_envs=64,
        envs_per_worker=1,
        envpool_batch_size=24,
        update_epochs=3,
        gamma=0.998,
        batch_size=2**15,
        batch_rows=128,
    )
    return args, make_sweep_config()
def all():
    '''All tested environments and platforms'''
    return {
        'pokemon_red': pokegym,
        'pokemon_red_pip': pokegym,
        'links_awaken': pokegym,
    }
def get_init_args(fn):
    sig = inspect.signature(fn)
    args = {}
    for name, param in sig.parameters.items():
        if name in ('self', 'env'):
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        else:
            args[name] = param.default if param.default is not inspect.Parameter.empty else None
    return args


def make_config(env) -> tuple[ModuleType, dict[str, Any], Namespace | None]:
    # TODO: Improve install checking with pkg_resources
    try:
        env_module = importlib.import_module(f'pufferlib.environments.{env}')
    except Exception as e:
        print(e)
        pufferlib.utils.install_requirements(env)
        env_module = importlib.import_module(f'pufferlib.environments.{env}')

    all_configs = all()
    args, sweep_config = all_configs[env]()

    env_kwargs = get_init_args(env_module.make_env)
    policy_kwargs = get_init_args(env_module.Policy.__init__)

    recurrent_kwargs = {}
    recurrent = env_module.Recurrent
    if recurrent is not None:
        recurrent_kwargs = dict(
            input_size=recurrent.input_size,
            hidden_size=recurrent.hidden_size,
            num_layers=recurrent.num_layers
        )

    return env_module, sweep_config, pufferlib.namespace(
        args=args,
        env_kwargs=env_kwargs,
        policy_kwargs=policy_kwargs,
        recurrent_kwargs=recurrent_kwargs,
    )


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
    args['args'] = CleanPuffeRL(**args['args'])
    args = pufferlib.namespace(**args)
    vec = "serial"
    assert vec in 'serial multiprocessing ray'.split()
    if vec == 'serial':
        args.vectorization = pufferlib.

    if args.sweep:
        args.track = True
    elif args.track:
        args.exp_name = init_wandb(args, env_module)


    assert sum((args.train, args.sweep, args.evaluate is not None)) == 1, 'Must specify exactly one of --train, --sweep, or --evaluate'
    if args.train:
        train(args, env_module)
            


if __name__ == '__main__':
    import pufferlib
    print(pufferlib.__version__)
    test_pokegym_map()