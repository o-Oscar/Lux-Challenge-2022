import numpy as np

from utils import teams
from utils.obs.base import BaseObsGenerator


class PositionIceFactoryObsGenerator(BaseObsGenerator):
    """
    Observations contain :
        - position of each robot
        - distance to each factory
        - position of ice
        - ice fullyness of each robot
    """

    def __init__(self, heavy_robot: bool = True):
        super().__init__()

        self.channel_nb = 6
        self.grid_channel_nb = 5
        self.vector_channel_nb = 1
        if heavy_robot:
            self.cargo_space = 1000
        else:
            self.cargo_space = 100

    def calc_obs(self, obs):
        # Pre_computation of the full grid features
        full_grid = np.zeros((self.channel_nb,) + obs["player_0"]["board"]["ice"].shape)

        # Positions of robots
        for i, team in enumerate(teams):
            for unit_name, unit in obs[team]["units"][team].items():
                full_grid[i, unit["pos"][0], unit["pos"][1]] = 1

        # Positions of factories
        for i, team in enumerate(teams):
            for factory in obs[team]["factories"][team].values():
                full_grid[2 + i][
                    factory["pos"][0] - 1 : factory["pos"][0] + 2,
                    factory["pos"][1] - 1 : factory["pos"][1] + 2,
                ] = 1

        # Position of ice
        full_grid[4] = obs["player_0"]["board"]["ice"]

        # Personal informations of the robots (for vector observations)
        for i, team in enumerate(teams):
            for unit_name, unit in obs[team]["units"][team].items():
                # Ice fullyness
                if unit["cargo"]["ice"] == self.cargo_space:
                    full_grid[5, unit["pos"][0], unit["pos"][1]] = 1

        # Invert the allied/opponent channels
        second_player_grid = full_grid.copy()
        second_player_grid[0] = full_grid[1]
        second_player_grid[1] = full_grid[0]
        second_player_grid[2] = full_grid[3]
        second_player_grid[3] = full_grid[2]

        return {teams[0]: full_grid, teams[1]: second_player_grid}
