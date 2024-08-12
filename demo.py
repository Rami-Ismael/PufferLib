from pdb import set_trace as T
import functools
import argparse
import shutil
import yaml
import uuid
import sys
import os


import pufferlib
import pufferlib.utils
import pufferlib.vector
import pufferlib.frameworks.cleanrl

from rich_argparse import RichHelpFormatter
from rich.traceback import install
from rich.console import Console
from rich import print as print
from rich.pretty import pprint as pprint

import clean_pufferl

import torch



def load_config(parser, config_path='config.yaml'):
    '''Just a fancy config loader. Populates argparse from
    yaml + env/policy fn signatures to give you a nice
    --help menu + some limited validation of the config'''
    args, _ = parser.parse_known_args()
    env_name, pkg_name = args.env, args.pkg

    with open(config_path) as f:
        config = yaml.safe_load(f)
    if 'default' not in config:
        raise ValueError('Deleted default config section?')
    if env_name not in config:
        raise ValueError(f'{env_name} not in config\n'
            'It might be available through a parent package, e.g.\n'
            '--config atari --env MontezumasRevengeNoFrameskip-v4.')

    default = config['default']
    env_config = config[env_name or pkg_name]
    pkg_name = pkg_name or env_config.get('package', env_name)
    pkg_config = config[pkg_name]
    # TODO: Check if actually installed
    env_module = pufferlib.utils.install_and_import(
        f'pufferlib.environments.{pkg_name}')
    make_name = env_config.get('env_name', None)
    make_env_args = [make_name] if make_name else []
    make_env = env_module.env_creator(*make_env_args)
    make_env_args = pufferlib.utils.get_init_args(make_env)
    policy_args = pufferlib.utils.get_init_args(env_module.Policy)
    rnn_args = pufferlib.utils.get_init_args(env_module.Recurrent)
    fn_sig = dict(env=make_env_args, policy=policy_args, rnn=rnn_args)
    config = vars(parser.parse_known_args()[0])

    valid_keys = 'env policy rnn train sweep'.split()
    for key in valid_keys:
        fn_subconfig = fn_sig.get(key, {})
        env_subconfig = env_config.get(key, {})
        pkg_subconfig = pkg_config.get(key, {})
        # Priority env->pkg->default->fn config
        config[key] = {**fn_subconfig, **default[key],
            **pkg_subconfig, **env_subconfig}

    for name in valid_keys:
        sub_config = config[name]
        for key, value in sub_config.items():
            data_key = f'{name}.{key}'
            cli_key = f'--{data_key}'.replace('_', '-')
            if isinstance(value, bool) and value is False:
                parser.add_argument(cli_key, default=value, action='store_true')
            elif isinstance(value, bool) and value is True:
                data_key = f'{name}.no_{key}'
                cli_key = f'--{data_key}'.replace('_', '-')
                parser.add_argument(cli_key, default=value, action='store_false')
            else:
                parser.add_argument(cli_key, default=value, type=type(value))

            config[name][key] = getattr(parser.parse_known_args()[0], data_key)
        config[name] = pufferlib.namespace(**config[name])

    pufferlib.utils.validate_args(make_env.func if isinstance(make_env, functools.partial) else make_env, config['env']) # This is where environments are created
    pufferlib.utils.validate_args(env_module.Policy, config['policy'])

    if 'use_rnn' in env_config:
        config['use_rnn'] = env_config['use_rnn']
    elif 'use_rnn' in pkg_config:
        config['use_rnn'] = pkg_config['use_rnn']
    else:
        config['use_rnn'] = default['use_rnn']

    parser.add_argument('--use_rnn', default=False, action='store_true',
        help='Wrap policy with an RNN')
    config['use_rnn'] = config['use_rnn'] or parser.parse_known_args()[0].use_rnn
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    parser.parse_args()
    wandb_name = make_name or env_name
    config['env_name'] = env_name
    config['exp_id'] = args.exp_id or args.env + '-' + str(uuid.uuid4())[:8]
    return wandb_name, pkg_name, pufferlib.namespace(**config), env_module, make_env, make_policy
   
def make_policy(env, env_module, args):
    policy = env_module.Policy(env, **args.policy)
    if args.use_rnn:
        policy = env_module.Recurrent(env, policy, **args.rnn)
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)

    return policy.to(args.train.device)

def init_wandb(args, name, id=None, resume=True):
    #os.environ["WANDB_SILENT"] = "true"
    import wandb
    wandb.require("core")
    wandb.init(
        id=id or wandb.util.generate_id(),
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config={
            'cleanrl': dict(args.train),
            'env': dict(args.env),
            'policy': dict(args.policy),
            #'recurrent': args.recurrent,
        },
        name=name,
        monitor_gym=True,
        save_code=True,
        resume=resume,
    )
    return wandb

def sweep(args, wandb_name, env_module, make_env):
    import wandb
    wandb.require("core")
    try:
        sweep_id = wandb.sweep(
            sweep=dict(args.sweep),
            project="pufferlib",
        )
    except Exception as e:
        print(f"The sweep is {args.sweep}")
    print(f"The sweep id is {sweep_id}")
    print(f"The sweep args is {args}")
    def main():
        try:
            args.exp_name = init_wandb(args, wandb_name, id=args.exp_id)
            # TODO: Add update method to namespace
            pprint(wandb.config.train)
            args.train.__dict__.update(dict(wandb.config.train))
            if args.policy.__dict__:
                args.policy.__dict__.update(dict(wandb.config.policy))
                pprint(f"The new configuration of policy is {args.policy.__dict__}")
            if args.env.__dict__:
                try:
                    args.env.__dict__.update(dict(wandb.config.env))
                    pprint(f"The new configuration of env is {args.env.__dict__}")
                except Exception as e:
                    print(f"The error is {e}")
                    pprint(f"What is the config of the env {wandb.config.env}")
                    pprint(f"What is the args.env__dict__ {args.env.__dict__}")
            args.track = True
            train(args, env_module, make_env)
        except Exception as e:
            import traceback
            traceback.print_exc()

    wandb.agent(sweep_id, main, count=400)
    #wandb.agent(
    #    sweep_id = "ajdb4yld" , 
    #    project = "pufferlib" , 
    #    function = main,
    #    count = 400
    #)
    

def train(args, env_module, make_env):
    args.wandb = None
    args.train.exp_id = args.exp_id
    if args.track:
        args.wandb = init_wandb(args, wandb_name, id=args.exp_id)

    vec = args.vec
    if vec == 'serial':
        vec = pufferlib.vector.Serial
    elif vec == 'multiprocessing':
        vec = pufferlib.vector.Multiprocessing
    elif vec == 'ray':
        vec = pufferlib.vector.Ray
    else:
        raise ValueError(f'Invalid --vector (serial/multiprocessing/ray).')
    print(f'The Environment Keyword Arguments')
    
    if args.train.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache() # I got OutOfMemoryError: CUDA out of memory
    vecenv = pufferlib.vector.make(
        make_env,
        env_kwargs=args.env,
        num_envs=args.train.num_envs,
        num_workers=args.train.num_workers,
        batch_size=args.train.env_batch_size,
        zero_copy=args.train.zero_copy,
        backend=vec,
    )
    policy = make_policy(vecenv.driver_env, env_module, args)
    train_config = args.train 
    train_config.track = args.track
    train_config.device = args.train.device
    train_config.env = args.env_name

   
    if args.backend == 'clean_pufferl':
        data = clean_pufferl.create(train_config, vecenv, policy, wandb=args.wandb)
        info = 0
        '''
        import time
        import torch
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as p:
        '''
        def early_stopping_base_on_ppo_loss_func(data):
            if train_config.update_epochs == 2:
                policy_loss_has_to_be_less_than =  0.003
                value_loss_has_to_be_greater_than = 0.00081819975093822
                approx_kl_has_to_be_less_than = 1.0
                clip_frac_has_to_be_greater_than = 0.002
                explain_variance_has_to_be_greater_than = 0.0
                if data.losses.policy_loss > policy_loss_has_to_be_less_than :
                    print(f"The policy loss is {data.losses.policy_loss}")
                    return False
                elif not ( data.losses.value_loss >= value_loss_has_to_be_greater_than or  data.losses.value_loss == 0.00):
                    print(f"The value loss need to be greater than {value_loss_has_to_be_greater_than} and not {data.losses.value_loss}")
                    return False
                elif data.losses.old_approx_kl <0:
                    print(f"The approx kl is {data.losses.old_approx_kl} and it should be greater than 0")
                    return False
                elif data.losses.explained_variance < explain_variance_has_to_be_greater_than:
                    print(f"The explained variance is {data.losses.explained_variance} and it should be greater than {explain_variance_has_to_be_greater_than}")
                    return False
                return   ( data.losses.value_loss >= value_loss_has_to_be_greater_than or data.losses.value_loss == 0.00) and  ( data.losses.approx_kl <= approx_kl_has_to_be_less_than or data.losses.approx_kl == 0.00) and ( data.losses.clipfrac >= clip_frac_has_to_be_greater_than or data.losses.clipfrac == 0.00)
            if train_config.update_epochs == 4:
                policy_loss_has_to_be_less_than = 0.019811158068478107
                value_loss_has_to_be_greater_than = 0.00081819975093822
                approx_kl_has_to_be_less_than = 1.0
                clip_frac_has_to_be_greater_than = 0.002
                approx_kl_has_to_be_greater_than = 0.001674623420854914
                if data.losses.policy_loss > policy_loss_has_to_be_less_than :
                    print(f"The policy loss is {data.losses.policy_loss}")
                    return False
                elif not ( data.losses.value_loss >= value_loss_has_to_be_greater_than or  data.losses.value_loss == 0.00):
                    print(f"The value loss need to be greater than {value_loss_has_to_be_greater_than} and not {data.losses.value_loss}")
                    return False
                elif data.losses.old_approx_kl <0:
                    print(f"The approx kl is {data.losses.old_approx_kl} and it should be greater than 0")
                    return False
                elif data.losses.approx_kl != 0 and data.losses.approx_kl < approx_kl_has_to_be_greater_than:
                    print(f"The approx kl is {data.losses.approx_kl} and it should be greater than {approx_kl_has_to_be_greater_than}")
                    return False
                elif data.losses.explained_variance < 0:
                    print(f"The explained variance is {data.losses.explained_variance} and it should be greater than 0")
                    return False
                return   ( data.losses.value_loss >= value_loss_has_to_be_greater_than or data.losses.value_loss == 0.00) and  ( data.losses.approx_kl <= approx_kl_has_to_be_less_than or data.losses.approx_kl == 0.00) and ( data.losses.clipfrac >= clip_frac_has_to_be_greater_than or data.losses.clipfrac == 0.00)
            if train_config.update_epochs == 8:
                policy_loss_has_to_be_less_than = 0.00537556956987828
                approx_kl_has_to_be_greater_than = 0.007167667066823924
                clipfrac_has_to_be_greater_than = 0.004302978515625
                if data.losses.policy_loss > policy_loss_has_to_be_less_than :
                    print(f"The policy loss is {data.losses.policy_loss}")
                    return False
                if data.losses.approx_kl != 0 and data.losses.approx_kl < approx_kl_has_to_be_greater_than:
                    print(f"The approx kl is {data.losses.approx_kl} and it should be greater than {approx_kl_has_to_be_greater_than}")
                    return False
                if data.losses.clipfrac != 0 and data.losses.clipfrac < clipfrac_has_to_be_greater_than:
                    print(f"The clipfrac is {data.losses.clipfrac} and it should be greater than {clipfrac_has_to_be_greater_than}")
                    return False
                return True
            if train_config.update_epochs == 16:
                policy_loss_has_to_be_less_than = 1.707766205072403e-04
                approx_kl_has_to_be_greater_than = 0.06862380067468621
                clipfrac_has_to_be_greater_than = 0.0788726806640625
                kl_has_to_be_less_than = 2
                value_loss_has_to_be_greater_than = 0.01902017235988751
                explain_variances_has_to_be_greater_than = 0.0
                if data.losses.value_loss !=0 and data.losses.value_loss < value_loss_has_to_be_greater_than:
                    print(f"The value loss is {data.losses.value_loss} and it should be greater than {value_loss_has_to_be_greater_than}")
                    return False
                if data.losses.policy_loss > policy_loss_has_to_be_less_than :
                    print(f"The policy loss is {data.losses.policy_loss}")
                    return False
                if data.losses.approx_kl != 0 and data.losses.approx_kl < approx_kl_has_to_be_greater_than:
                    print(f"The approx kl is {data.losses.approx_kl} and it should be greater than {approx_kl_has_to_be_greater_than}")
                    return False
                if data.losses.approx_kl != 0 and data.losses.approx_kl > kl_has_to_be_less_than:
                    print(f"The approx kl is {data.losses.approx_kl} and it should be less than {kl_has_to_be_less_than}")
                    return False
                if data.losses.clipfrac != 0 and data.losses.clipfrac < clipfrac_has_to_be_greater_than:
                    print(f"The clipfrac is {data.losses.clipfrac} and it should be greater than {clipfrac_has_to_be_greater_than}")
                    return False
                if data.losses.explained_variance < explain_variances_has_to_be_greater_than:
                    print(f"The explained variance is {data.losses.explained_variance} and it should be greater than {explain_variances_has_to_be_greater_than}")
                    return False
                return True
            try:
                if data.stats is not None and  data.stats.keys() and data.epoch >= 40 and  data.stats["number_of_uniqiue_coordinate_it_explored"]<=670:
                    return False
                if data.stats is not None and  data.stats.keys() and data.epoch >= 40 and  data.stats["number_of_uniqiue_coordinate_it_explored"]<=670:
                    return False
                if data.stats is not None and  data.stats.keys() and data.epoch >= 150 and  data.stats["number_of_uniqiue_coordinate_it_explored"]<=1400:
                    print(f"The number of unique coordinate it explored is {data.stats['number_of_uniqiue_coordinate_it_explored']} and it should be greater than {1400}")
                    return False
                if data.stats is not None and  data.stats.keys() and data.epoch >= 200 and  data.stats["number_of_uniqiue_coordinate_it_explored"]<=1700:
                    print(f"The number of unique coordinate it explored is {data.stats['number_of_uniqiue_coordinate_it_explored']} and it should be greater than {1700}")
                    return False
            except Exception as e:
                print(f"The error is {e}")
                T()
            return True
            
        while data.global_step < args.train.total_timesteps and early_stopping_base_on_ppo_loss_func(data):
            try:
                clean_pufferl.evaluate(data)
                clean_pufferl.train(data)
            except KeyboardInterrupt:
                clean_pufferl.close(data)
                os._exit(0)
            except Exception:
                Console().print_exception()
                os._exit(0)

        clean_pufferl.evaluate(data)
        clean_pufferl.close(data)

    elif args.backend == 'sb3':
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.common.env_util import make_vec_env
        from sb3_contrib import RecurrentPPO

        envs = make_vec_env(lambda: make_env(**args.env),
            n_envs=args.train.num_envs, seed=args.train.seed, vec_env_cls=DummyVecEnv)

        model = RecurrentPPO("CnnLstmPolicy", envs, verbose=1,
            n_steps=args.train.batch_rows*args.train.bptt_horizon,
            batch_size=args.train.batch_size, n_epochs=args.train.update_epochs,
            gamma=args.train.gamma
        )

        model.learn(total_timesteps=args.train.total_timesteps)

if __name__ == '__main__':
    install(show_locals=False) # Rich tracebacks
    # TODO: Add check against old args like --config to demo
    parser = argparse.ArgumentParser(
            description=f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]'
        ' demo options. Shows valid args for your env and policy',
        formatter_class=RichHelpFormatter, add_help=False)
    assert 'config' not in sys.argv, '--config deprecated. Use --env'
    parser.add_argument('--env', '--environment', type=str,
        default='squared', help='Name of specific environment to run')
    parser.add_argument('--pkg', '--package', type=str, default=None, help='Configuration in config.yaml to use')
    parser.add_argument('--backend', type=str, default='clean_pufferl', help='Train backend (clean_pufferl, sb3)')
    parser.add_argument('--mode', type=str, default='train', choices='train eval evaluate sweep autotune baseline profile'.split())
    parser.add_argument('--eval-model-path', type=str, default=None, help='Path to model to evaluate')
    parser.add_argument('--baseline', action='store_true', help='Baseline run')
    parser.add_argument('--no-render', action='store_true', help='Disable render during evaluate')
    parser.add_argument('--vec', '--vector', '--vectorization', type=str,
        default='serial', choices='serial multiprocessing ray'.split())
    parser.add_argument('--exp-id', '--exp-name', type=str, default=None, help="Resume from experiment")
    parser.add_argument('--wandb-entity', type=str, default='compress_rl', help='WandB entity')
    parser.add_argument('--wandb-project', type=str, default='pufferlib', help='WandB project')
    parser.add_argument('--wandb-group', type=str, default='debug', help='WandB group')
    parser.add_argument('--track', action='store_true', help='Track on WandB')
    wandb_name, pkg, args, env_module, make_env, make_policy = load_config(parser)
    pprint( args.__dict__ )
    print(f"The arguments of of the trains is")
    pprint(args.train.__dict__)
    print(f"The number of step of each agents does is {args.train.batch_size / args.train.num_envs}")
    agent_steps:int = args.train.batch_size / args.train.num_envs
    print(f"The number of steps which maybe which be random because of entropy coeficient is  { agent_steps * args.train.ent_coef } ")

    if args.baseline:
        assert args.mode in ('train', 'eval', 'evaluate')
        args.track = True
        version = '.'.join(pufferlib.__version__.split('.')[:2])
        args.exp_id = f'puf-{version}-{args.env_name}'
        args.wandb_group = f'puf-{version}-baseline'
        shutil.rmtree(f'experiments/{args.exp_id}', ignore_errors=True)
        run = init_wandb(args, args.exp_id, resume=False)
        if args.mode in ('eval', 'evaluate'):
            model_name = f'puf-{version}-{args.env_name}_model:latest'
            artifact = run.use_artifact(model_name)
            data_dir = artifact.download()
            model_file = max(os.listdir(data_dir))
            args.eval_model_path = os.path.join(data_dir, model_file)

    if args.mode == 'train':
        train(args, env_module, make_env)
    elif args.mode in ('eval', 'evaluate'):
        try:
            clean_pufferl.rollout(
                make_env,
                args.env,
                agent_creator=make_policy,
                agent_kwargs={'env_module': env_module, 'args': args},
                model_path=args.eval_model_path,
                device=args.train.device
            )
        except KeyboardInterrupt:
            os._exit(0)
    elif args.mode == 'sweep':
        sweep(args, wandb_name, env_module, make_env)
    elif args.mode == 'autotune':
        pufferlib.vector.autotune(make_env, batch_size=args.train.env_batch_size)
    elif args.mode == 'profile':
        import cProfile
        cProfile.run('train(args, env_module, make_env)', 'stats.profile')
        import pstats
        from pstats import SortKey
        p = pstats.Stats('stats.profile')
        p.sort_stats(SortKey.TIME).print_stats(10)
    elif args.mode == 'evaluate' and pkg == 'pokemon_red':
        import pokemon_red_eval
        pokemon_red_eval.rollout(
            make_env,
            args.env,
            agent_creator=make_policy,
            agent_kwargs={'env_module': env_module, 'args': args},
            model_path=args.eval_model_path,
            device=args.train.device,
        )
