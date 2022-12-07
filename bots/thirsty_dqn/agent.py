import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions.categorical import Categorical

from learning.dqn import BaseAgent
from utils.env import Env

BIG_NUMBER = 1e10


class Agent(BaseAgent):
    def __init__(self, env: Env, device):
        super().__init__()
        self.env = env
        self.device = device

        n_obs = self.env.obs_generator.channel_nb
        n_act = self.env.action_handler.action_nb

        self.q_network = nn.Sequential(
            nn.Conv2d(n_obs + n_act, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding="same"),
        )

        self.v_network = nn.Sequential(
            nn.Conv2d(n_obs, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding="same"),
        )

    def refine_actions(self, obs, init_action, mask, tau):
        mask_np = mask.detach().cpu().numpy()[0]
        possible_actions = np.where(mask_np)
        all_test_actions = []
        for a, x, y in zip(*possible_actions):
            cur_action = th.clone(init_action)
            cur_action[0, x, y] = a
            all_test_actions.append(cur_action)

        all_test_actions = th.concat(all_test_actions, dim=0)
        rep_obs = obs.repeat((all_test_actions.shape[0], 1, 1, 1))
        rep_mask = mask.repeat((all_test_actions.shape[0], 1, 1, 1))
        q_values = self.q_eval(rep_obs, all_test_actions, rep_mask)
        q_values = q_values.detach().cpu().numpy() / tau

        to_return = th.clone(init_action)
        sum = np.zeros((48, 48))
        for i, (a, x, y) in enumerate(zip(*possible_actions)):
            sum[x, y] = sum[x, y] + np.exp(q_values[i, x, y])

        full_rand = np.random.random(size=(48, 48))
        for i, (a, x, y) in enumerate(zip(*possible_actions)):
            full_rand[x, y] = full_rand[x, y] - np.exp(q_values[i, x, y]) / sum[x, y]
            if full_rand[x, y] <= 0:
                to_return[0, x, y] = a
                full_rand[x, y] = 100000

        # for x, y in zip(*np.where(mask_np[0])):
        #     print(to_return[0, x, y])
        # exit()

        return to_return

    def sample_actions(self, obs, masks, tau):
        """
        I have to do smart stuff here to get a kinda good action.
        Let's start by sampling actions one by one for each robot
        """

        if obs.shape[0] > 1:
            to_return = []
            for o, m in zip(obs, masks):
                o = o.view((1, 11, 48, 48))
                m = m.view((1, 5, 48, 48))
                to_return.append(self.sample_actions(o, m, tau))
            return th.concat(to_return, dim=0)

        cur_action_th = th.zeros((1, 48, 48), dtype=th.long, device=self.device)
        # we sould repeat this line a few times. But let's not go there yet
        cur_action_th = self.refine_actions(obs, cur_action_th, masks, tau)
        return cur_action_th

    def q_eval(self, obs, act, masks):
        hot_act = nn.functional.one_hot(
            act.type(th.long), num_classes=self.env.action_handler.action_nb
        ).type(th.float32)
        hot_act = th.permute(hot_act, (0, 3, 1, 2)) * masks

        inp = th.concat([obs, hot_act], dim=1)
        return self.q_network(inp).view((inp.shape[0], inp.shape[2], inp.shape[3]))

    def v_eval(self, obs):
        return self.v_network(obs).view((obs.shape[0], obs.shape[2], obs.shape[3]))

    def to_one_hot(self, act):
        return (
            nn.functional.one_hot(
                act.type(th.long), num_classes=self.env.action_handler.action_nb
            )
            .type(th.float32)
            .permute((0, 3, 1, 2))
        )
