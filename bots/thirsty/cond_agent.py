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


class MaskedConvolution(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, mask_center=False, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, **kwargs)
        self.init_mask(kernel_size, mask_center)

    def forward(self, x):
        self.conv.weight.data *= self.mask
        return self.conv(x)

    def init_mask(self, kernel_size):
        raise NotImplementedError


class VerticalStackConvolution(MaskedConvolution):
    def init_mask(self, kernel_size, mask_center):
        mask = th.ones(kernel_size, kernel_size)
        mask[kernel_size // 2 + 1 :, :] = 0

        if mask_center:
            mask[kernel_size // 2, :] = 0

        self.register_buffer("mask", mask)


class HorizontalStackConvolution(MaskedConvolution):
    def init_mask(self, kernel_size, mask_center):
        mask = th.zeros(kernel_size, kernel_size)
        mask[kernel_size // 2, : kernel_size // 2 + 1] = 1

        if mask_center:
            mask[kernel_size // 2, kernel_size // 2] = 0

        self.register_buffer("mask", mask)


class PixelCnn(nn.Module):
    def __init__(self, c_in, c_out, n_layers):
        super().__init__()

        v_modules = [
            VerticalStackConvolution(c_in, 8, 5, mask_center=True, padding="same")
        ]
        h_modules = [
            HorizontalStackConvolution(c_in, 8, 5, mask_center=True, padding="same")
        ]

        for i in range(n_layers - 2):
            v_modules.append(
                VerticalStackConvolution(8, 8, 1, mask_center=True, padding="same")
            )
            h_modules.append(
                HorizontalStackConvolution(8, 8, 1, mask_center=True, padding="same")
            )

        v_modules.append(
            VerticalStackConvolution(8, c_out, 1, mask_center=True, padding="same")
        )
        h_modules.append(
            HorizontalStackConvolution(8, c_out, 1, mask_center=True, padding="same")
        )

        self.v_stack = nn.ModuleList(v_modules)
        self.h_stack = nn.ModuleList(h_modules)

    def forward(self, inp):
        vx = inp
        hx = inp
        for v, h in zip(self.v_stack, self.h_stack):
            vx = th.relu(v(vx))
            hx = vx + th.relu(h(hx))
        return hx


class GridHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(6, 32, 11, padding="same")

    def forward(self, obs):
        return th.relu(self.backbone(obs))


class VectorHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(5, 32, 1, padding="same")

    def forward(self, obs):
        return th.relu(self.backbone(obs))


class FullHead(nn.Module):
    def __init__(self, action_nb, out_size):
        super().__init__()
        self.action_nb = action_nb
        self.out_size = out_size

        self.action_head = PixelCnn(5, 32, 2)
        self.grid_head = GridHead()
        self.vector_head = VectorHead()

        self.backbone = PixelCnn(32 * 3, self.out_size, 2)

    def forward(self, obs, actions):
        grid_obs = obs[:, :6]
        vector_obs = obs[:, 6:]
        actions_hot = (
            nn.functional.one_hot(actions.type(th.long), num_classes=self.action_nb)
            .permute(0, 3, 1, 2)
            .type(th.float32)
        )

        action_feat = self.action_head(actions_hot)
        grid_feat = self.grid_head(grid_obs)
        vector_feat = self.vector_head(vector_obs)
        full_feat = th.concat([action_feat, grid_feat, vector_feat], dim=1)
        return self.backbone(full_feat)


class CondAgent(BaseAgent):
    def __init__(self, env: Env):
        super().__init__()
        self.env = env
        action_nb = env.action_handler.action_nb
        self.critic = FullHead(action_nb, 1)
        self.actor = FullHead(action_nb, action_nb)

    def get_value(self, obs, action):
        return self.critic(obs, action).squeeze(dim=1)

    def get_action(self, obs, masks, action=None, device=None):

        if action is None:
            action = self.sample(obs, masks, device)

        logits = self.actor(obs, action)
        logits = logits * masks - BIG_NUMBER * (1 - masks)
        logits = logits.permute(0, 2, 3, 1)
        probs = Categorical(logits=logits)

        # action = th.argmax(logits, dim=-1) * 0
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(obs, action).squeeze(dim=1),
        )

    def sample(self, obs, mask, device):

        print("sampling")
        action_shape = (
            obs.shape[0],
            obs.shape[2],
            obs.shape[3],
        )

        # create the return object
        actions = th.zeros(action_shape, dtype=th.long, device=device)

        # for all pixels in the image : run the model, sample, and write the action time the mask
        for j in range(action_shape[1]):
            print(j)
            for i in range(action_shape[2]):
                logits = self.actor(obs, actions)
                logits = logits * mask - BIG_NUMBER * (1 - mask)
                probs = Categorical(logits=logits[:, :, j, i])
                cur_mask = th.max(mask[:, :, j, i], dim=1)[0]
                actions[:, j, i] = probs.sample() * cur_mask
        print("done sampling !!")

        return actions
