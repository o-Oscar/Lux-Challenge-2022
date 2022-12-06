import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions.categorical import Categorical

from learning.dqn import BaseAgent
from utils.env import Env

BIG_NUMBER = 1e10


# class ObsHead(nn.Module):
#     def __init__(self, env: Env):
#         super().__init__()
#         self.grid_head = nn.Conv2d(6, 32, 11, padding="same")
#         self.vector_head = nn.Conv2d(5, 32, 1, padding="same")

#     def forward(self, obs):
#         grid_obs = obs[:, :6]
#         vector_obs = obs[:, 6:]

#         grid_feat = self.grid_head(grid_obs)
#         vector_feat = self.vector_head(vector_obs)

#         return th.relu(th.concat([grid_feat, vector_feat], dim=1))


def apply_qt(inp, t):
    e = th.exp(-t)
    f = 1 - e
    s = th.mean(inp, dim=1, keepdim=True)
    return e * inp + f * s


class Agent(BaseAgent):
    def __init__(self, env: Env, device):
        super().__init__()
        self.env = env
        self.device = device

        n_obs = self.env.obs_generator.channel_nb
        n_act = self.env.action_handler.action_nb

        # self.actor = nn.Sequential(
        #     nn.Conv2d(n_obs + n_act + 1, 64, 3, padding="same"),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 3, padding="same"),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 3, padding="same"),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 3, padding="same"),
        #     nn.ReLU(),
        #     nn.Conv2d(64, n_act, 3, padding="same"),
        # )

        self.q_network = nn.Sequential(
            nn.Conv2d(n_obs + n_act, 64, 3, padding="same"),
            nn.ReLU(),
            # nn.Conv2d(64, 64, 3, padding="same"),
            # nn.ReLU(),
            # nn.Conv2d(64, 64, 3, padding="same"),
            # nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding="same"),
        )

        self.actor = nn.Sequential(
            nn.Conv2d(n_obs + n_act + 1, n_act, 3, padding="same"),
        )

        self.all_x0 = []
        for i in range(self.env.action_handler.action_nb):
            x0 = np.zeros((1, self.env.action_handler.action_nb, 48, 48))
            x0[0, i] = 1
            self.all_x0.append(th.tensor(x0, dtype=th.float32, device=self.device))

    def cat_sample(self, probs):
        probs = Categorical(probs=th.permute(probs, (0, 2, 3, 1)))
        to_return = nn.functional.one_hot(
            probs.sample(), num_classes=self.env.action_handler.action_nb
        ).type(th.float32)
        to_return = th.permute(to_return, (0, 3, 1, 2))
        return to_return

    def sample_actions(self, obs, masks):
        """
        Diffusion process
        """
        all_t = np.logspace(-2, 0, num=10)

        action_shape = (
            obs.shape[0],
            self.env.action_handler.action_nb,
            obs.shape[2],
            obs.shape[3],
        )
        ones = th.ones(action_shape).to(self.device)
        robot_mask = th.max(masks, dim=1, keepdims=True).values
        action_nb = th.sum(masks, dim=1, keepdims=True)
        action_nb = action_nb * robot_mask + self.env.action_handler.action_nb * (
            1 - robot_mask
        )
        complete_mask = masks + ones * (1 - robot_mask)

        first_prob = complete_mask / action_nb
        cur_a = self.cat_sample(first_prob)

        for i in range(len(all_t) - 1, -1, -1):
            if i == 0:
                dt = all_t[i]
            else:
                dt = all_t[i] - all_t[i - 1]

            t = th.tensor(all_t[i], dtype=th.float32, device=self.device)
            dt = th.tensor(dt, dtype=th.float32, device=self.device)

            logits = self.calc_logits(obs, cur_a * masks, t)
            logits = logits * masks - BIG_NUMBER * (1 - masks)
            probs = th.softmax(logits, dim=1)

            new_x_probs = 0
            for x0 in self.all_x0:
                p_x0 = th.sum(probs * x0, dim=1, keepdims=True)
                sum = th.sum(x0 * apply_qt(cur_a, t), dim=1, keepdim=True)
                new_x_probs = (
                    new_x_probs
                    + p_x0 * (apply_qt(cur_a, dt) * apply_qt(x0, t - dt)) / sum
                )
            cur_a = self.cat_sample(new_x_probs)

        # return cur_a
        return Categorical(probs=th.permute(cur_a, (0, 2, 3, 1))).sample()

    def calc_logits(self, obs, act, t):
        t_inp = th.ones(
            (act.shape[0], 1, act.shape[2], act.shape[3]), device=self.device
        ) * th.log(t)

        inp = th.concat([obs, act, t_inp], dim=1)
        logits = self.actor(inp) * 0
        return logits

    def q_eval(self, obs, act, masks):
        hot_act = nn.functional.one_hot(
            act.type(th.long), num_classes=self.env.action_handler.action_nb
        ).type(th.float32)
        hot_act = th.permute(hot_act, (0, 3, 1, 2)) * masks

        inp = th.concat([obs, hot_act], dim=1)
        return self.q_network(inp).view((inp.shape[0], inp.shape[2], inp.shape[3]))
