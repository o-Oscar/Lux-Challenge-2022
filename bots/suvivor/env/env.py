import gym
from gym import spaces
import numpy as np

import luxai2022
from luxai_runner.utils import to_json
from pathlib import Path
import json


class Env(gym.Env):
    def __init__(self):
        self.observation_shape = (2 * 3,)
        self.observation_space = spaces.Box(
            low=-np.ones(self.observation_shape),
            high=np.ones(self.observation_shape),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(6)

        self.env = luxai2022.LuxAI2022()

    def step_env(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        change_obs = self.env.state.get_change_obs(self.state_obs)
        self.state_obs = obs["player_0"]

        self.replay["observations"].append(change_obs)
        self.replay["actions"].append(actions)

    def save_replay(self):
        self.replay = to_json(self.replay)
        with open("replay_custom.json", "w") as f:
            json.dump(self.replay, f)

    def reset(self):

        obs = self.env.reset()
        self.state_obs = self.env.state.get_compressed_obs()

        self.replay = dict(observations=[], actions=[], dones=[], rewards=[])
        self.replay["observations"].append(self.state_obs)

        # bid phase
        actions = {
            "player_0": {"faction": "AlphaStrike", "bid": 0},
            "player_1": {"faction": "MotherMars", "bid": 0},
        }
        self.step_env(actions)

        # factory positionning phase # TODO : find a better
        for i in range(obs["player_0"]["board"]["factories_per_team"]):
            actions = {
                "player_0": {
                    "spawn": obs["player_0"]["board"]["spawns"]["player_0"][i * 3],
                    "metal": self.env.env_cfg.INIT_WATER_METAL_PER_FACTORY,
                    "water": self.env.env_cfg.INIT_WATER_METAL_PER_FACTORY,
                },
                "player_1": {
                    "spawn": obs["player_1"]["board"]["spawns"]["player_1"][i * 3],
                    "metal": self.env.env_cfg.INIT_WATER_METAL_PER_FACTORY,
                    "water": self.env.env_cfg.INIT_WATER_METAL_PER_FACTORY,
                },
            }
            self.step_env(actions)

        # one more turn of factory placement
        self.step_env({"player_0": {}, "player_1": {}})

        # turn 0 : creating one robot for each team
        actions = {
            "player_0": {list(self.env.state.factories["player_0"].keys())[0]: 0},
            "player_1": {list(self.env.state.factories["player_1"].keys())[0]: 0},
        }
        self.step_env(actions)

        # computation of ore positions
        self.all_ore_positions = np.array(np.where(self.env.state.board.ore > 0)).T
        self.all_ice_positions = np.array(np.where(self.env.state.board.ice > 0)).T
        players = ["player_0", "player_1"]
        main_factory_names = [
            list(self.env.state.factories[player].keys())[0] for player in players
        ]
        self.main_robots_names = [
            list(self.env.state.units[player].keys())[0] for player in players
        ]
        self.main_factory_positions = [
            self.env.state.factories[player][name]
            for player, name in zip(players, main_factory_names)
        ]

    def find_closest(self, pos, all_pos):
        return all_pos[np.argmin(np.sum(np.abs(all_pos - pos), axis=1), axis=0)]

    def get_scripted_move_action(self, delta):
        if delta[0] > 0:
            return np.array([0, 3, 0, 0, 0])
        if delta[0] < 0:
            return np.array([0, 1, 0, 0, 0])
        if delta[1] > 0:
            return np.array([0, 2, 0, 0, 0])
        if delta[1] < 0:
            return np.array([0, 4, 0, 0, 0])

    def get_scripted_action(self):
        unit = self.env.state.units["player_1"][self.main_robots_names[1]]
        if unit.power < 10:
            return {}

        pos = np.array([unit.pos.y, unit.pos.x])
        delta = self.find_closest(pos, self.all_ice_positions) - pos
        if delta[0] == 0 and delta[1] == 0:
            return {self.main_robots_names[1]: np.array([3, 0, 0, 0, 0])}

        scripted_action = self.get_scripted_move_action(delta)

        return {self.main_robots_names[1]: scripted_action}

    def get_main_action(self, a):
        return {}

    def step(self, a):

        actions = {
            "player_0": self.get_main_action(a),
            "player_1": self.get_scripted_action(),
        }
        obs, rewards, dones, infos = self.env.step(actions)
        change_obs = self.env.state.get_change_obs(self.state_obs)
        self.state_obs = obs["player_0"]

        self.replay["observations"].append(change_obs)
        self.replay["actions"].append(actions)

        # return custom_obs, custom_reward, custom_done, {}
