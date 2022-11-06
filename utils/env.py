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
        self.teams = ["player_0", "player_1"]

        self.robot_to_env_actions = [
            None,  # ne rien faire
            np.array([0, 1, 0, 0, 0]),  # bouger en haut
            np.array([0, 2, 0, 0, 0]),  # bouger à droite
            np.array([0, 3, 0, 0, 0]),  # bouger en bas
            np.array([0, 4, 0, 0, 0]),  # bouger à gauche
            np.array([1, 0, 0, 100, 0]),  # transférer de la glace
            np.array([1, 0, 1, 100, 0]),  # transférer des minerais
            np.array([2, 0, 4, 50, 0]),  # transférer des minerais
            np.array([2, 0, 4, 100, 0]),  # transférer des minerais
            np.array([2, 0, 4, 150, 0]),  # transférer des minerais
            np.array([3, 0, 0, 0, 0]),  # creuser
        ]

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
            for team in self.teams:
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
        self.env.step(self.factory_actions())

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

        for team in self.teams:
            for factory_name, factory in self.env.state.factories[team].items():
                if (
                    factory.power >= light_config.POWER_COST * power_fac
                    and factory.cargo.metal >= light_config.METAL_COST
                ):
                    actions[team][factory_name] = 0

        return actions

    def add_robots_actions(self, action_dict, robots_actions):
        for team in self.teams:
            for unit_name, unit in self.env.state.units[team].items():
                cur_action = self.robot_to_env_actions[robots_actions[team][unit_name]]
                if cur_action is not None:
                    action_dict[team][unit_name] = cur_action

    def step(self, a):
        actions = self.factory_actions()
        self.add_robots_actions(actions, a)

        obs, rewards, dones, infos = self.env.step(actions)
        return obs, rewards, dones, infos

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
