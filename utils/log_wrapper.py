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
        # TODO : find a way to dissable this usefull feature
        # if self.compressed_log or self.full_log:
        #     self.save()

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