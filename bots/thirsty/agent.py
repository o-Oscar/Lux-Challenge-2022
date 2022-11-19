import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions.categorical import Categorical

from learning.ppo import BaseAgent
from utils.env import Env

BIG_NUMBER = 1e10


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer


class ObsHead(nn.Module):
    def __init__(self, env: Env):
        super().__init__()
        self.grid_head = nn.Conv2d(6, 32, 11, padding="same")
        self.vector_head = nn.Conv2d(5, 32, 1, padding="same")

    def forward(self, obs):
        grid_obs = obs[:, :6]
        vector_obs = obs[:, 6:]

        grid_feat = self.grid_head(grid_obs)
        vector_feat = self.vector_head(vector_obs)

        return th.relu(th.concat([grid_feat, vector_feat], dim=1))


class Agent(BaseAgent):
    def __init__(self, env: Env, use_multi_path=True):
        super().__init__()
        if use_multi_path:
            self.critic = nn.Sequential(
                ObsHead(env),
                layer_init(nn.Conv2d(64, 64, 1, padding="same")),
                nn.Tanh(),
                layer_init(nn.Conv2d(64, 1, 1, padding="same"), std=1.0),
            )
            self.actor = nn.Sequential(
                ObsHead(env),
                layer_init(nn.Conv2d(64, 64, 1, padding="same")),
                nn.Tanh(),
                layer_init(
                    nn.Conv2d(64, env.action_handler.action_nb, 1, padding="same"),
                    std=0.01,
                ),
            )
        else:
            self.critic = nn.Sequential(
                layer_init(nn.Conv2d(9, 64, 1, padding="same")),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 1, padding="same")),
                nn.Tanh(),
                layer_init(nn.Conv2d(64, 1, 1, padding="same"), std=1.0),
            )
            self.actor = nn.Sequential(
                layer_init(nn.Conv2d(9, 64, 1, padding="same")),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 1, padding="same")),
                nn.Tanh(),
                layer_init(
                    nn.Conv2d(64, env.action_handler.action_nb, 1, padding="same"),
                    std=0.01,
                ),
            )

    def get_value(self, obs):
        return self.critic(obs).squeeze(dim=1)

    def get_action(self, obs, masks, action=None):
        logits = self.actor(obs)
        logits = logits * masks - BIG_NUMBER * (1 - masks)
        logits = logits.permute(0, 2, 3, 1)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        # action = th.argmax(logits, dim=-1) * 0
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(obs).squeeze(dim=1),
        )
