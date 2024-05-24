import torch
from torch import nn

import pufferlib.models
import pufferlib.pytorch

class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=256, hidden_size=256, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

@torch.compiler.disable
def decode_map(codes):
    codes = codes.unsqueeze(1).long()
    factors = [4, 4, 16, 5, 3, 5, 5, 6, 7, 4]
    n_channels = sum(factors)
    obs = torch.zeros(codes.shape[0], n_channels, 11, 15, device='cuda')

    add, div = 0, 1
    # TODO: check item/tier order
    for mod in factors:
        obs.scatter_(1, add+(codes//div)%mod, 1)
        add += mod
        div *= mod

    return obs

class Policy(pufferlib.models.Policy):
    def __init__(self, env, *args, framestack):
        super().__init__(env)
        self.emulated = env.emulated

        self.num_actions = self.action_space.n
        #self.num_actions = self.action_space.shape[0]
        hidden_size = 256
        output_size = 256

        self.map_2d = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(59, 64, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(128, hidden_size//2)),
            nn.ReLU(),
        )

        self.embed = nn.Embedding(128, 32)
        self.player_1d = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(32*44, hidden_size//2)),
            nn.ReLU(),
        )

        self.proj = nn.Linear(hidden_size, output_size)

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(output_size, self.num_actions), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)

        self.ob_map = torch.zeros(100000, 59, 11, 15, device='cuda')
        self.players = torch.zeros(100000, 44, device='cuda').int()

        self.player_emb = torch.zeros(100000, 128, device='cuda')
        self.map_emb = torch.zeros(100000, 128, device='cuda')


    def encode_observations(self, observations, unflatten=True):
        batch = observations.shape[0]
        if unflatten:
            x = pufferlib.pytorch.nativize_observation(observations, self.emulated)
        else:
            x = observations

        with torch.no_grad():
            ob_map = decode_map(x['map']).float()

        player = x['player']
        #ob_map = self.ob_map[:batch]
        #player = self.players[:batch]

        ob_map = self.map_2d(ob_map)
        #ob_map = self.map_emb[:batch]

        ob_player = self.embed(player)
        #ob_player = self.embed(x['player'].int())
        ob_player = ob_player.flatten(1)
        ob_player = self.player_1d(ob_player)

        ob = torch.cat([ob_map, ob_player], dim=1)
        return self.proj(ob), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        #action = [self.actor(flat_hidden)]
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
