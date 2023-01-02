import gym
from gym import spaces
import numpy as np
from scipy import ndimage
import luxai2022
from luxai_runner.utils import to_json
from pathlib import Path
import json
import pickle
import time
import luxai2022.config
import luxai2022.state
from utils import teams
from utils.action.base import BaseActionHandler
from utils.obs.base import BaseObsGenerator
from utils.reward.base import BaseRewardGenerator
from utils.log_wrapper import LogWrapper

import matplotlib.pyplot as plt
import matplotlib

DEFAULT_LOG_PATH = Path("results/logs/")


class Env(gym.Env):
    def __init__(
        self,
        action_hanlder: BaseActionHandler,
        obs_generator: BaseObsGenerator,
        reward_generator: BaseRewardGenerator,
        power_cost: bool = False,
        heavy_robot: bool = True,
        water_consumption: int = 1,
        max_length: int = 1100,
    ):
        robots = luxai2022.config.EnvConfig().ROBOTS

        if not power_cost:
            robots["LIGHT"].MOVE_COST = 0
            robots["LIGHT"].DIG_COST = 0
            robots["LIGHT"].INIT_POWER = 10000
            robots["LIGHT"].BATTERY_CAPACITY = 10000

            robots["HEAVY"].MOVE_COST = 0
            robots["HEAVY"].RUBBLE_MOVEMENT_COST = 0
            robots["HEAVY"].DIG_COST = 0
            robots["HEAVY"].INIT_POWER = 10000
            robots["HEAVY"].BATTERY_CAPACITY = 10000

        self.env = LogWrapper(
            luxai2022.LuxAI2022(
                validate_action_space=False,
                verbose=0,
                ROBOTS=robots,
                FACTORY_WATER_CONSUMPTION=water_consumption,
                max_episode_length=max_length,
                NUM_WEATHER_EVENTS_RANGE=[0, 0],
            )
        )

        self.max_length = max_length
        self.action_handler = action_hanlder
        self.obs_generator = obs_generator
        self.reward_generator = reward_generator
        self.heavy_robot = heavy_robot

    def calc_unit_pos(self, obs):
        to_return = {team: {} for team in teams}
        for team in teams:
            for unit_id, unit in obs[team]["units"][team].items():
                to_return[team][unit_id] = unit["pos"]
        return to_return

    def reset(self, **kwargs):

        obs = self.env.reset(**kwargs)

        # bid phase
        actions = {
            "player_0": {"faction": "AlphaStrike", "bid": 0},
            "player_1": {"faction": "MotherMars", "bid": 0},
        }
        self.env.step(actions)

        # factory positionning phase

        n_factories = obs["player_0"]["board"]["factories_per_team"]
        valid_pos = np.where(obs["player_0"]["board"]["valid_spawns_mask"])
        n_valid = int(np.sum(obs["player_0"]["board"]["valid_spawns_mask"]))
        ice_grid = obs["player_0"]["board"]["ice"]
        fac_pos = []

        # 4 kernels to count the number of ice closer than 4 different distances
        # (there is not always enough place to only use the most little distance)
        nb_kernels = 4
        ice_scores = []
        for k in range(1, nb_kernels + 1):
            distance = 10 * k + 1
            kernel = np.zeros((2 * distance, 2 * distance))
            # put ones where we are closer to the center than distance
            for i in range(2 * distance):
                for j in range(2 * distance):
                    if np.abs(i - distance) + np.abs(j - distance) < distance:
                        kernel[i, j] = 1
            convolute_image = ndimage.convolve(
                ice_grid, kernel, mode="constant", cval=0.0
            )
            ice_scores.append(convolute_image)

        # sort the id with the 4 scores
        sorted_poses_per_distance = []
        for num_distance, ice_score in enumerate(ice_scores):
            pos_ids = list(range(n_valid))
            pos_ids.sort(
                key=lambda pos_id: ice_score[
                    valid_pos[0][pos_id], valid_pos[1][pos_id]
                ],
                reverse=True,
            )
            for i in range(n_valid):
                if num_distance < nb_kernels - 1 and (
                    ice_score[
                        valid_pos[0][pos_ids[i]],
                        valid_pos[1][pos_ids[i]],
                    ]
                    < 4
                ):
                    pos_ids = pos_ids[:i]
                    break
            sorted_poses_per_distance.append(pos_ids)

        # choose the id for the factories, prioritizing the first distances
        num_distance = 0
        while len(fac_pos) < n_factories * 2:
            pos_id = sorted_poses_per_distance[num_distance][0]
            fac_pos.append((valid_pos[0][pos_id], valid_pos[1][pos_id]))
            to_delete_pos_ids = []
            for remaining_pos_id in sorted_poses_per_distance[-1]:
                if (
                    remaining_pos_id not in to_delete_pos_ids
                    and np.linalg.norm(
                        np.array(
                            [
                                valid_pos[0][remaining_pos_id],
                                valid_pos[1][remaining_pos_id],
                            ]
                        )
                        - np.array([valid_pos[0][pos_id], valid_pos[1][pos_id]]),
                        ord=np.inf,
                    )
                    < 8
                ):
                    to_delete_pos_ids.append(remaining_pos_id)

            for to_delete_pos_id in to_delete_pos_ids:
                for sorted_poses in sorted_poses_per_distance:
                    if to_delete_pos_id in sorted_poses:
                        sorted_poses.remove(to_delete_pos_id)

            while len(sorted_poses_per_distance[num_distance]) == 0:
                num_distance += 1

        for i in range(n_factories):
            for j, team in enumerate(teams):
                actions = {t: {} for t in teams}
                # [print(k) for k, v in obs[team]["board"].items()]
                # import matplotlib.pyplot as plt

                # plt.imshow(obs[team]["board"]["valid_spawns_mask"])
                # plt.show()
                # exit()
                actions[team] = {
                    "spawn": list(fac_pos[2 * i + j]),
                    "metal": self.env.env_cfg.INIT_WATER_METAL_PER_FACTORY,
                    "water": self.env.env_cfg.INIT_WATER_METAL_PER_FACTORY,
                }
                self.env.step(actions)

        # one more turn of factory placement
        # self.env.step({"player_0": {}, "player_1": {}})

        # turn 0 : creating robots for each team
        obs, rewards, dones, infos = self.env.step(self.factory_actions())

        unit_obs = self.obs_generator.calc_obs(obs)
        action_masks = self.action_handler.calc_masks(obs)
        units_pos = self.calc_unit_pos(obs)

        self.old_obs = obs
        return unit_obs, action_masks, units_pos, n_factories

    def factory_actions(self):
        """
        Create a heavy/light robot when possible. Takes into account factory power, metal cargo and weather.
        """
        actions = {
            "player_0": {},
            "player_1": {},
        }

        if self.heavy_robot:
            config = self.env.env_cfg.ROBOTS["HEAVY"]
            action = 1
        else:
            config = self.env.env_cfg.ROBOTS["LIGHT"]
            action = 0

        if self.env.state.weather_schedule[self.env.state.real_env_steps] == 2:
            power_fac = self.env_cfg.WEATHER["COLD_SNAP"]["POWER_CONSUMPTION"]
        else:
            power_fac = 1

        for team in teams:
            for factory_name, factory in self.env.state.factories[team].items():
                if (
                    factory.power >= config.POWER_COST * power_fac
                    and factory.cargo.metal >= config.METAL_COST
                ):
                    actions[team][factory_name] = action

        return actions

    def fuse_actions(self, *action_dicts):
        to_return = {team: {} for team in teams}
        for action_dict in action_dicts:
            for team in teams:
                to_return[team].update(action_dict[team])
        return to_return

    def step(self, a):
        factory_actions = self.factory_actions()
        units_actions = self.action_handler.network_to_robots(self.old_obs, a)
        actions = self.fuse_actions(factory_actions, units_actions)

        obs, rewards, dones, infos = self.env.step(actions)
        done = dones["player_0"] or dones["player_1"]

        unit_obs = self.obs_generator.calc_obs(obs)
        action_masks = self.action_handler.calc_masks(obs)
        rewards, rewards_monotoring = self.reward_generator.calc_rewards(
            self.old_obs, actions, obs
        )

        units_pos = self.calc_unit_pos(obs)

        self.old_obs = obs
        return unit_obs, rewards, rewards_monotoring, action_masks, done, units_pos

    def save(self, **kwargs):
        return self.env.save(**kwargs)

    @property
    def state(self) -> luxai2022.state.State:
        return self.env.state

    @property
    def env_cfg(self) -> luxai2022.config.EnvConfig:
        return self.env.env_cfg

    def reset_draw(self, **kwargs):
        nb_game = 7
        nb_kernels = 4
        fig, ax = plt.subplots(
            nb_game, nb_kernels + 3, figsize=(7, 10), sharex=True, sharey=True
        )
        for num_row in range(nb_game):
            for num_col in range(nb_kernels + 3):
                ax[num_row][num_col].axis("off")

        for num_fig in range(nb_game):
            obs = self.env.reset(**kwargs)

            # bid phase
            actions = {
                "player_0": {"faction": "AlphaStrike", "bid": 0},
                "player_1": {"faction": "MotherMars", "bid": 0},
            }
            self.env.step(actions)

            # factory positionning phase

            n_factories = obs["player_0"]["board"]["factories_per_team"]
            valid_pos = np.where(obs["player_0"]["board"]["valid_spawns_mask"])
            n_valid = int(np.sum(obs["player_0"]["board"]["valid_spawns_mask"]))
            ice_grid = obs["player_0"]["board"]["ice"]
            fac_pos = []

            # 4 kernels to count the number of ice closer than 4 different distances
            # (there is not always enough place to only use the most little distance)
            nb_kernels = 4
            ice_scores = []
            ax[num_fig][0].imshow(ice_grid)
            for k in range(1, nb_kernels + 1):
                distance = 10 * k + 1
                kernel = np.zeros((2 * distance, 2 * distance))
                # put ones where we are closer to the center than distance
                for i in range(2 * distance):
                    for j in range(2 * distance):
                        if np.abs(i - distance) + np.abs(j - distance) < distance:
                            kernel[i, j] = 1
                convolute_image = ndimage.convolve(
                    ice_grid, kernel, mode="constant", cval=0.0
                )
                ice_scores.append(convolute_image)
                ax[num_fig][k].imshow(convolute_image)

            # sort the id with the 4 scores
            sorted_poses_per_distance = []
            for num_distance, ice_score in enumerate(ice_scores):
                pos_ids = list(range(n_valid))
                pos_ids.sort(
                    key=lambda pos_id: ice_score[
                        valid_pos[0][pos_id], valid_pos[1][pos_id]
                    ],
                    reverse=True,
                )
                for i in range(n_valid):
                    if num_distance < nb_kernels - 1 and (
                        ice_score[
                            valid_pos[0][pos_ids[i]],
                            valid_pos[1][pos_ids[i]],
                        ]
                        < 4
                    ):
                        pos_ids = pos_ids[:i]
                        break
                sorted_poses_per_distance.append(pos_ids)

            # choose the id for the factories, prioritizing the first distances
            num_distance = 0
            while len(fac_pos) < n_factories * 2:
                pos_id = sorted_poses_per_distance[num_distance][0]
                fac_pos.append((valid_pos[0][pos_id], valid_pos[1][pos_id]))
                to_delete_pos_ids = []
                for remaining_pos_id in sorted_poses_per_distance[-1]:
                    if (
                        remaining_pos_id not in to_delete_pos_ids
                        and np.linalg.norm(
                            np.array(
                                [
                                    valid_pos[0][remaining_pos_id],
                                    valid_pos[1][remaining_pos_id],
                                ]
                            )
                            - np.array([valid_pos[0][pos_id], valid_pos[1][pos_id]]),
                            ord=np.inf,
                        )
                        < 8
                    ):
                        to_delete_pos_ids.append(remaining_pos_id)

                for to_delete_pos_id in to_delete_pos_ids:
                    for sorted_poses in sorted_poses_per_distance:
                        if to_delete_pos_id in sorted_poses:
                            sorted_poses.remove(to_delete_pos_id)

                while len(sorted_poses_per_distance[num_distance]) == 0:
                    num_distance += 1

            cmap = matplotlib.cm.get_cmap("viridis")
            ice_grid = ice_grid.astype(float)
            ice_grid_rgba_img = cmap(ice_grid)
            ice_grid_rgb_img = np.delete(ice_grid_rgba_img, 3, 2)

            facto_position_img = np.copy(ice_grid_rgb_img)
            for start_pos in fac_pos:
                facto_position_img[
                    start_pos[0] - 1 : start_pos[0] + 2,
                    start_pos[1] - 1 : start_pos[1] + 2,
                ] = [1, 0, 0]

            ax[num_fig][nb_kernels + 1].imshow(facto_position_img)

            old_fac_pos = []
            for i in range(n_factories * 2):
                pos_id = int(n_valid * (i + 1) / (n_factories * 2 + 1))
                old_fac_pos.append((valid_pos[0][pos_id], valid_pos[1][pos_id]))

            facto_position_img = np.copy(ice_grid_rgb_img)
            for start_pos in old_fac_pos:
                facto_position_img[
                    start_pos[0] - 1 : start_pos[0] + 2,
                    start_pos[1] - 1 : start_pos[1] + 2,
                ] = [1, 0, 0]

            ax[num_fig][nb_kernels + 2].imshow(facto_position_img)

        plt.show()
        exit()


def get_env() -> Env:
    return Env()
