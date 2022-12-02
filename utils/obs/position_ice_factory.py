import numpy as np
from utils import teams
import matplotlib.pyplot as plt
from utils.obs.base import BaseObsGenerator


class PositionIceFactoryObsGenerator(BaseObsGenerator):
    """
    Observations contain :
        - position of each robot
        - distance to each factory
        - position of ice
        - ice of each robot
        - ice fullyness of each robot
    """

    def __init__(self, heavy_robot: bool = True):
        super().__init__()

        self.channel_nb = 7
        self.grid_channel_nb = 7
        self.vector_channel_nb = 0
        if heavy_robot:
            self.cargo_space = 1000
        else:
            self.cargo_space = 100

    def calc_obs(self, obs):
        # pre_computation of the full grid features
        full_grid = np.zeros((self.channel_nb,) + obs["player_0"]["board"]["ice"].shape)

        # robot positions
        for i, team in enumerate(teams):
            for unit_name, unit in obs[team]["units"][team].items():
                full_grid[i, unit["pos"][1], unit["pos"][0]] = 1

        # delta to factories
        all_x = np.arange(obs["player_0"]["board"]["ice"].shape[0])
        all_y = np.arange(obs["player_0"]["board"]["ice"].shape[1])
        all_x, all_y = np.meshgrid(all_x, all_y)
        for i, team in enumerate(teams):
            all_deltas = []
            for factory in obs[team]["factories"][team].values():
                cur_delta = np.abs(all_x - factory["pos"][0]) + np.abs(
                    all_y - factory["pos"][1]
                )
                all_deltas.append(cur_delta)
            if len(all_deltas) > 0:
                full_grid[2 + i] = np.min(all_deltas, axis=0)

        # ice
        full_grid[4] = obs["player_0"]["board"]["ice"]

        # robot ice cargot
        for i, team in enumerate(teams):
            for unit_name, unit in obs[team]["units"][team].items():
                # ice fullyness
                if unit["cargo"]["ice"] == self.cargo_space:
                    full_grid[5, unit["pos"][1], unit["pos"][0]] = 1

                # ice
                full_grid[6, unit["pos"][1], unit["pos"][0]] = (
                    unit["cargo"]["ice"] / self.cargo_space
                )

        # invert the allied/opponent channels
        second_player_grid = full_grid.copy()
        second_player_grid[0] = full_grid[1]
        second_player_grid[1] = full_grid[0]
        second_player_grid[2] = full_grid[3]
        second_player_grid[3] = full_grid[2]

        return {teams[0]: full_grid, teams[1]: second_player_grid}
