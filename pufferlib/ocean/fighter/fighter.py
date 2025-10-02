import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.fighter import binding

PLAYER_OBS_N = 27

class Fighter(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=1,
            num_agents=2,
            human_agent_idx=0,
            buf = None, seed=0):

        # env
        self.num_agents = num_envs*num_agents
        self.render_mode = render_mode
        self.report_interval = report_interval
        
        self.num_obs = 27
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(self.num_obs,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(5)

        super().__init__(buf=buf)
        c_envs = []
        for i in range(num_envs):
            env_id = binding.env_init(
                self.observations[i*num_agents:(i+1)*num_agents],
                self.actions[i*num_agents:(i+1)*num_agents],
                self.rewards[i*num_agents:(i+1)*num_agents],
                self.terminals[i*num_agents:(i+1)*num_agents],
                self.truncations[i*num_agents:(i+1)*num_agents],
                i + seed * num_envs,
                num_agents=num_agents,
                human_agent_idx=human_agent_idx
            )
            c_envs.append(env_id)

        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1

        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)
        
    def close(self):
        binding.vec_close(self.c_envs)

def load_moves():
    import os
    from pathlib import Path

    #create the binaries directory
    binary_dir = Path("resources/fighter/binaries")
    binary_dir.mkdir(parents=True, exist_ok=True)

    # Path to move set data
    data_dir = Path("resources/fighter/paul-moves")

    # Get all npz files
    npz_files = sorted(data_dir.glob("*.npz"))

    # Process each file
    for i, character_path in enumerate(npz_files):
        binary_file = "paul.bin"
        binary_path = binary_dir / binary_file
        move_data = np.load(character_path, allow_pickle=True)
        joints = move_data["joints"]
        print(character_path)
        print(joints.shape)
        print("Frames: ", joints.shape[0])
        print("Joints: ", joints.shape[1])
        print("XYZ per joint: ", joints.shape[2])

        print("Pelvis frame 0:", joints[0,0])
        #save_move_binary(move_data, str(binary_path))

def test_performance(timeout=10, atn_cache=1024):
    num_envs=1000;
    env = Fighter(num_envs=num_envs)
    env.reset()
    tick = 0

    actions = np.random.randint(0, env.single_action_space.n, (atn_cache, 5*num_envs))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    sps = num_envs * tick / (time.time() - start)
    print(f'SPS: {sps:,}')
if __name__ == '__main__':
    #test_performance()
    load_moves()                         

