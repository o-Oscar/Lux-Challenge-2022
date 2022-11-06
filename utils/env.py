import gym
from gym import spaces
import numpy as np

import luxai2022
from luxai_runner.utils import to_json
from pathlib import Path
import json
import pickle
import time
import luxai2022.config
import luxai2022.state
from utils import teams
from utils.obs.default import DefaultObsGenerator
from utils.action.default import DefaultActionHandler
from utils.reward.default import DefaultRewardGenerator

DEFAULT_LOG_PATH = Path("results/logs/")


class LogWrapper(gym.Wrapper):
    def __init__(self, env: luxai2022.LuxAI2022, log_path=DEFAULT_LOG_PATH):
        self.env = env
        self.log_path = log_path

        self.action_space = spaces.Discrete(6)

        self.compressed_log = {}
        self.full_log = {}

    def step(self, action):
        obs, rewards, dones, infos = self.env.step(action)
        change_obs = self.env.state.get_change_obs(self.state_obs)
        self.state_obs = obs["player_0"]

        self.compressed_log["observations"].append(change_obs)
        self.compressed_log["actions"].append(action)

        self.full_log["observations"].append(obs)
        self.full_log["actions"].append(action)

        return obs, rewards, dones, infos

    def reset(self, **kwargs):
        if self.compressed_log or self.full_log:
            self.save()

        obs = self.env.reset()

        self.state_obs = self.env.state.get_compressed_obs()
        self.compressed_log = dict(observations=[], actions=[])
        self.compressed_log["observations"].append(self.state_obs)

        self.full_log = dict(observations=[obs], actions=[])

        return obs

    def save(self, full_save=True, convenient_save=True):
        if full_save:
            timestr = time.strftime("%Y_%m_%d-%H:%M:%S")
            compressed_path = self.log_path / "compressed" / (timestr + ".json")
            full_path = self.log_path / "full" / (timestr + ".pkl")

            compressed_path.parent.mkdir(exist_ok=True, parents=True)
            full_path.parent.mkdir(exist_ok=True, parents=True)

            self.compressed_log = to_json(self.compressed_log)

            with open(compressed_path, "w") as f:
                json.dump(self.compressed_log, f)
            with open(full_path, "wb") as f:
                pickle.dump(self.full_log, f)

        if convenient_save:
            self.compressed_log = to_json(self.compressed_log)
            with open("replay_custom.json", "w") as f:
                json.dump(self.compressed_log, f)

    @property
    def state(self) -> luxai2022.state.State:
        return self.env.state

    @property
    def env_cfg(self) -> luxai2022.config.EnvConfig:
        return self.env.env_cfg


class Env(gym.Env):
    def __init__(self):
        self.observation_shape = (2 * 3,)
        self.observation_space = spaces.Box(
            low=-np.ones(self.observation_shape),
            high=np.ones(self.observation_shape),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(6)

        self.env = LogWrapper(luxai2022.LuxAI2022())

        self.obs_generator = DefaultObsGenerator()
        self.action_handler = DefaultActionHandler()
        self.reward_generator = DefaultRewardGenerator()

    def reset(self):

        obs = self.env.reset()

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
        self.old_obs = obs

        return unit_obs

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
        units_actions = self.action_handler.network_to_robots(a)
        actions = self.fuse_actions(factory_actions, units_actions)

        obs, rewards, dones, infos = self.env.step(actions)
        unit_obs = self.obs_generator.calc_obs(obs)
        action_masks = self.action_handler.calc_masks(obs)
        rewards = self.reward_generator.calc_rewards(self.old_obs, actions, obs)

        self.old_obs = obs
        return unit_obs, rewards, action_masks
        # return unit_obs, action_masks

    def save(self, **kwargs):
        return self.env.save(**kwargs)

    @property
    def state(self) -> luxai2022.state.State:
        return self.env.state

    @property
    def env_cfg(self) -> luxai2022.config.EnvConfig:
        return self.env.env_cfg


def get_env():
    return Env()
