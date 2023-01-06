import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions.categorical import Categorical

from utils.agent.base import BaseAgent
from utils.obs.base import BaseObsGenerator
from utils.action.base import BaseActionHandler

BIG_NUMBER = 1e10


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer


class ObsHead(nn.Module):
    def __init__(
        self,
        obs_generator: BaseObsGenerator,
        grid_kernel_size=11,
        grid_layers_nb=1,
        grid_post_channel_nb=32,
        vector_post_channel_nb=32,
        use_relu=False,
    ):
        super().__init__()
        self.grid_channel_nb = obs_generator.grid_channel_nb
        self.vector_channel_nb = obs_generator.vector_channel_nb

        if self.grid_channel_nb == 0:
            print("No grid channel given")
            raise ValueError

        grid_layers = [
            nn.Conv2d(
                self.grid_channel_nb,
                grid_post_channel_nb,
                grid_kernel_size,
                padding="same",
            ),
        ]

        for _ in range(grid_layers_nb - 1):
            if use_relu:
                grid_layers.append(nn.ReLU())
            else:
                grid_layers.append(nn.Tanh())

            grid_layers.append(
                nn.Conv2d(
                    grid_post_channel_nb,
                    grid_post_channel_nb,
                    grid_kernel_size,
                    padding="same",
                )
            )

        self.grid_head = nn.Sequential(*grid_layers)

        if self.vector_channel_nb > 0:
            self.vector_head = nn.Conv2d(
                self.vector_channel_nb, vector_post_channel_nb, 1, padding="same"
            )
        else:
            self.vector_head = None

    def forward(self, obs):
        grid_obs = obs[:, : self.grid_channel_nb]
        vector_obs = obs[:, self.grid_channel_nb :]

        grid_feat = self.grid_head(grid_obs)

        if self.vector_head is not None:
            vector_feat = self.vector_head(vector_obs)
            return th.relu(th.concat([grid_feat, vector_feat], dim=1))

        else:
            return th.relu(grid_feat)


class ConvAgent(BaseAgent):
    def __init__(
        self,
        obs_generator: BaseObsGenerator,
        action_handler: BaseActionHandler,
        grid_kernel_size=11,
        grid_layers_nb=1,
        grid_post_channel_nb=32,
        vector_post_channel_nb=32,
        inside_kernel_size=1,
        inside_layers_nb=1,
        final_kernel_size=1,
        final_layers_nb=1,
        use_relu=False,
    ):
        super().__init__()

        inside_dim = grid_post_channel_nb + vector_post_channel_nb
        self.inside_dim = inside_dim
        self.grid_kernel_size = grid_kernel_size
        self.grid_layers_nb = grid_layers_nb
        self.grid_post_channel_nb = grid_post_channel_nb
        self.vector_post_channel_nb = vector_post_channel_nb
        self.inside_kernel_size = inside_kernel_size
        self.inside_layers_nb = inside_layers_nb
        self.final_kernel_size = final_kernel_size
        self.final_layers_nb = final_layers_nb
        self.use_relu = use_relu

        # Observation Head
        critic_layers = [
            ObsHead(
                obs_generator,
                grid_kernel_size,
                grid_layers_nb,
                grid_post_channel_nb,
                vector_post_channel_nb,
                use_relu,
            )
        ]
        actor_layers = [
            ObsHead(
                obs_generator,
                grid_kernel_size,
                grid_layers_nb,
                grid_post_channel_nb,
                vector_post_channel_nb,
                use_relu,
            )
        ]

        # Inside Network
        for _ in range(inside_layers_nb):
            critic_layers.append(
                layer_init(
                    nn.Conv2d(
                        inside_dim, inside_dim, inside_kernel_size, padding="same"
                    )
                )
            )
            if use_relu:
                critic_layers.append(nn.ReLU())
            else:
                critic_layers.append(nn.Tanh())

            actor_layers.append(
                layer_init(
                    nn.Conv2d(
                        inside_dim, inside_dim, inside_kernel_size, padding="same"
                    )
                )
            )
            if use_relu:
                actor_layers.append(nn.ReLU())
            else:
                actor_layers.append(nn.Tanh())

        # Final Network
        critic_layers.append(
            layer_init(
                nn.Conv2d(inside_dim, 1, final_kernel_size, padding="same"), std=1.0
            )
        )
        actor_layers.append(
            layer_init(
                nn.Conv2d(
                    inside_dim,
                    action_handler.action_nb,
                    final_kernel_size,
                    padding="same",
                ),
                std=0.01,
            )
        )

        for _ in range(final_layers_nb - 1):
            if use_relu:
                critic_layers.append(nn.ReLU())
            else:
                critic_layers.append(nn.Tanh())
            critic_layers.append(
                layer_init(nn.Conv2d(1, 1, final_kernel_size, padding="same"))
            )

            if use_relu:
                actor_layers.append(nn.ReLU())
            else:
                actor_layers.append(nn.Tanh())
            actor_layers.append(
                layer_init(
                    nn.Conv2d(
                        action_handler.action_nb,
                        action_handler.action_nb,
                        final_kernel_size,
                        padding="same",
                    )
                )
            )

        self.critic = nn.Sequential(*critic_layers)
        self.actor = nn.Sequential(*actor_layers)

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
