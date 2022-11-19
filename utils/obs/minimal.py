import matplotlib.pyplot as plt
import numpy as np

from utils import teams
from utils.obs.base import BaseObsGenerator


class MinimalObsGenerator(BaseObsGenerator):
    """
    Observations contain :
        - ice position
        - ore position
        - position of each robot
        - position of each factory
        - ore of each robot
        - ice of each robot
        - power of each robot
        - time in the day
    """

    def __init__(self):
        super().__init__()

        self.channel_nb = 9

    def calc_obs(self, obs):
        # pre_computation of the full grid features
        full_grid = np.zeros((self.channel_nb,) + obs["player_0"]["board"]["ice"].shape)

        # ice
        full_grid[0] = obs["player_0"]["board"]["ice"]

        # ore
        full_grid[1] = obs["player_0"]["board"]["ore"]

        # unit pos
        for team in teams:
            for unit in obs[team]["units"][team].values():
                full_grid[2, unit["pos"][1], unit["pos"][0]] = 1

        # factory pos
        for team in teams:
            for factory in obs[team]["factories"][team].values():
                full_grid[3, factory["pos"][1], factory["pos"][0]] = 1

        # robot specific features
        for i, team in enumerate(teams):
            for unit_name, unit in obs[team]["units"][team].items():

                # ice
                ice_feat = unit["cargo"]["ice"] / 100
                ore_feat = unit["cargo"]["ore"] / 100
                power_feat = unit["power"] / 150
                full_grid[4, unit["pos"][1], unit["pos"][0]] = ice_feat
                full_grid[5, unit["pos"][1], unit["pos"][0]] = ore_feat
                full_grid[6, unit["pos"][1], unit["pos"][0]] = power_feat

        # time in the day
        full_grid[7] = np.sin(np.pi * 2 * obs[team]["real_env_steps"] / 50)
        full_grid[8] = np.cos(np.pi * 2 * obs[team]["real_env_steps"] / 50)

        # invert the allied/opponent channels
        second_player_grid = full_grid.copy()

        return {teams[0]: full_grid, teams[1]: second_player_grid}
