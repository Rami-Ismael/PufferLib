import runpy
import os
import sys
from rich import print
from pydantic import BaseModel
import optuna
from typing import Callable, Dict, List, Optional

'''
def test_use_sde_perforamnces(script: str) -> None:
    num_seeds = 5
    for use_sde in [True, False]:
        params: list[str] = [""]
        print(f"Testing use_sde = {use_sde}")
        if use_sde:
            params = params + [f"--use-sde"]
        params = params + [f"--headless"]
        print(f"params: {params}")
        for seed in range(num_seeds):
            sys.argv = params + [f"--seed={seed}"]
            runpy.run_path(path_name=script, run_name="__main__")


def check_if_reward_the_agent_with_a_normalize_money_amount(script: str) -> None:
    num_seeds = 5
    for reward_scale in range(1, 4):
        for reward_the_agent_with_a_normalize_money_amount in [True, False]:
            params: list[str] = [""]
            print(f"Testing reward_the_agent_with_a_normalize_money_amount = {reward_the_agent_with_a_normalize_money_amount}")
            if reward_the_agent_with_a_normalize_money_amount:
                params = params + ["--reward-for-money-amount"]
            params = params + ["--headless"] + [f"--reward-scale={reward_scale}"]
            print(f"params: {params}")
            for seed in range(1, num_seeds):
                sys.argv = params + [f"--seed={seed}"]
                runpy.run_path(path_name=script, run_name="__main__")
'''
class Tuner(BaseModel):
    script: str
    num_seeds: int
    params_fn: Callable[[optuna.Trial], Dict]
    seed: int
    def run(self):
        for seed in range(self.num_seeds):
            sys.argv = self.params + [f"--seed={seed}"]
            runpy.run_path(path_name=self.script, run_name="__main__")

def check_if_annealing_helps(script: str) -> None:
    num_seeds = 5
    for annealing in [False, True]:
        params: list[str] = [""]
        print(f"Testing annealing = {annealing}")
        if  not annealing:
            params = params + [f"---args.no-anneal-lr"]
        params = params + [f"--headless"]
        print(f"params: {params}")
        for seed in range(num_seeds):
            sys.argv = params + [f"--seed={seed}"]
            runpy.run_path(path_name=script, run_name="__main__")

def check_if_reward_agent_number_of_unique_caught_pokemon(script):

if __name__ == "__main__":
    print(os.getcwd())
    #check_if_reward_the_agent_with_a_normalize_money_amount("baselines/baseline_fast_v2.py")
    check_if_annealing_helps("demo.py")