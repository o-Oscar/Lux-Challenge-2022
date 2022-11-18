import numpy as np
from utils import teams
import matplotlib.pyplot as plt
from utils.obs.base import BaseObsGenerator


class PositionTimeObsGenerator(BaseObsGenerator):
    """
    Observations contain :
        - time
        - position of each robot
    """

    def __init__(self):
        super().__init__()

        self.channel_nb = 4

    def calc_obs(self, obs):
        # pre_computation of the full grid features
        full_grid = np.zeros((self.channel_nb,) + obs["player_0"]["board"]["ice"].shape)

        # robot specific features
        for i, team in enumerate(teams):
            for unit_name, unit in obs[team]["units"][team].items():
                full_grid[i, unit["pos"][1], unit["pos"][0]] = 1

        # time in the day
        full_grid[2] = np.sin(np.pi * 2 * obs[team]["real_env_steps"] / 50)
        full_grid[3] = np.cos(np.pi * 2 * obs[team]["real_env_steps"] / 50)

        # invert the allied/opponent channels
        second_player_grid = full_grid.copy()
        second_player_grid[0] = full_grid[1]
        second_player_grid[1] = full_grid[0]

        return {teams[0]: full_grid, teams[1]: second_player_grid}
