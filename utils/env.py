import json
import pickle
import time
from pathlib import Path

import gym
import luxai2022
import luxai2022.config
import luxai2022.state
import numpy as np
from gym import spaces
from luxai_runner.utils import to_json

from utils import teams
from utils.action.base import BaseActionHandler
from utils.log_wrapper import LogWrapper
from utils.obs.base import BaseObsGenerator
from utils.reward.base import BaseRewardGenerator

DEFAULT_LOG_PATH = Path("results/logs/")


class Env(gym.Env):
    def __init__(
        self,
        action_hanlder: BaseActionHandler,
        obs_generator: BaseObsGenerator,
        reward_generator: BaseRewardGenerator,
        **kwargs
    ):
        self.env = LogWrapper(
            luxai2022.LuxAI2022(validate_action_space=False, verbose=0, **kwargs)
        )
        self.action_handler = action_hanlder
        self.obs_generator = obs_generator
        self.reward_generator = reward_generator

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
        for i in range(n_factories):
            actions = {}
            for team in teams:
                spawns = obs[team]["board"]["spawns"][team]
                m, M = np.min(spawns, axis=0), np.max(spawns, axis=0)
                delta = M - m
                main_axis = np.argmax(delta)
                main_size = np.max(delta)
                main_pos = int((i + 1) / (n_factories + 1) * main_size)
                sub_pos = int(delta[1 - main_axis] / 2)
                pos = m + np.array(
                    [main_pos, sub_pos] if main_axis == 0 else [sub_pos, main_pos]
                )
                actions[team] = {
                    "spawn": list(pos),
                    "metal": self.env.env_cfg.INIT_WATER_METAL_PER_FACTORY,
                    "water": self.env.env_cfg.INIT_WATER_METAL_PER_FACTORY,
                }
            self.env.step(actions)

        # one more turn of factory placement
        self.env.step({"player_0": {}, "player_1": {}})

        # turn 0 : creating robots for each team
        obs, rewards, dones, infos = self.env.step(self.factory_actions())

        unit_obs = self.obs_generator.calc_obs(obs)
        action_masks = self.action_handler.calc_masks(obs)
        units_pos = self.calc_unit_pos(obs)

        self.old_obs = obs
        return unit_obs, action_masks, units_pos

    def factory_actions(self):
        """
        Create a light robot when possible. Takes into account factory power, metal cargo and weather.
        """
        actions = {
            "player_0": {},
            "player_1": {},
        }

        light_config = self.env.env_cfg.ROBOTS["LIGHT"]

        if self.env.state.weather_schedule[self.env.state.real_env_steps] == 2:
            power_fac = self.env_cfg.WEATHER["COLD_SNAP"]["POWER_CONSUMPTION"]
        else:
            power_fac = 1

        for team in teams:
            for factory_name, factory in self.env.state.factories[team].items():
                if (
                    factory.power >= light_config.POWER_COST * power_fac
                    and factory.cargo.metal >= light_config.METAL_COST
                ):
                    actions[team][factory_name] = 0

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
        rewards = self.reward_generator.calc_rewards(self.old_obs, actions, obs)

        units_pos = self.calc_unit_pos(obs)

        self.old_obs = obs
        return unit_obs, rewards, action_masks, done, units_pos

    def save(self, **kwargs):
        return self.env.save(**kwargs)

    @property
    def state(self) -> luxai2022.state.State:
        return self.env.state

    @property
    def env_cfg(self) -> luxai2022.config.EnvConfig:
        return self.env.env_cfg


def get_env() -> Env:
    return Env()
