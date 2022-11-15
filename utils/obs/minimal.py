import numpy as np
from utils import teams
import matplotlib.pyplot as plt
from utils.obs.base import BaseObsGenerator


class MinimalObsGenerator(BaseObsGenerator):
    """
    Observations contain :
        - ice position
        - ore position
        - position of each robot
        - ore of each robot
        - ice of each robot
        - power of each robot
    """

    def __init__(self):
        super().__init__()

        self.channel_nb = 11

    def calc_obs(self, obs):
        # pre_computation of the full grid features
        full_grid = np.zeros((self.channel_nb,) + obs["player_0"]["board"]["ice"].shape)

        # ice
        full_grid[0] = obs["player_0"]["board"]["ice"]

        # ore
        full_grid[1] = obs["player_0"]["board"]["ore"]

        # robot specific features
        for i, team in enumerate(teams):
            for unit_name, unit in obs[team]["units"][team].items():

                # ice
                ice_feat = unit["cargo"]["ice"] / 100
                ore_feat = unit["cargo"]["ore"] / 100
                power_feat = unit["power"] / 150
                full_grid[2 + i, unit["pos"][1], unit["pos"][0]] = 1
                full_grid[3, unit["pos"][1], unit["pos"][0]] = ice_feat
                full_grid[4, unit["pos"][1], unit["pos"][0]] = ore_feat
                full_grid[5, unit["pos"][1], unit["pos"][0]] = power_feat

        # invert the allied/opponent channels
        second_player_grid = full_grid.copy()
        second_player_grid[2] = full_grid[3]
        second_player_grid[3] = full_grid[2]

        return {teams[0]: full_grid, teams[1]: second_player_grid}
