import numpy as np
from utils import teams
import matplotlib.pyplot as plt
from utils.obs import ObsGenerator

OBS_GRID_CHANNELS = 10


class DefaultObsGenerator(ObsGenerator):
    def __init__(self):
        super().__init__()

    def calc_obs(self, obs):
        # pre_computation of the full grid features
        full_grid = np.zeros(
            (OBS_GRID_CHANNELS,) + obs["player_0"]["board"]["ice"].shape
        )

        # ice
        full_grid[0] = obs["player_0"]["board"]["ice"]

        # ore
        full_grid[1] = obs["player_0"]["board"]["ore"]

        # factory
        for team in teams:
            for factory in obs[team]["factories"][team].values():
                full_grid[2, factory["pos"][1], factory["pos"][0]] = 1

        # delta
        all_x = np.arange(obs["player_0"]["board"]["ice"].shape[0])
        all_y = np.arange(obs["player_0"]["board"]["ice"].shape[1])
        all_x, all_y = np.meshgrid(all_x, all_y)
        all_deltas = []
        for team in teams:
            for factory in obs[team]["factories"][team].values():
                cur_delta = np.abs(all_x - factory["pos"][0]) + np.abs(
                    all_y - factory["pos"][1]
                )
                all_deltas.append(cur_delta)
        full_grid[3] = np.min(all_deltas, axis=0)

        # time in the day
        full_grid[4] = np.sin(np.pi * 2 * obs[team]["real_env_steps"] / 50)
        full_grid[5] = np.cos(np.pi * 2 * obs[team]["real_env_steps"] / 50)

        # robot specific features
        for team in teams:
            for unit_name, unit in obs[team]["units"][team].items():

                # ice
                ice_feat = unit["cargo"]["ice"] / 100
                ore_feat = unit["cargo"]["ore"] / 100
                power_feat = unit["power"] / 150
                full_grid[6, unit["pos"][1], unit["pos"][0]] = 1
                full_grid[7, unit["pos"][1], unit["pos"][0]] = ice_feat
                full_grid[8, unit["pos"][1], unit["pos"][0]] = ore_feat
                full_grid[9, unit["pos"][1], unit["pos"][0]] = power_feat

        return full_grid
