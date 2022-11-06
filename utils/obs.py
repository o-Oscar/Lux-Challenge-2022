import numpy as np
from utils import teams
import matplotlib.pyplot as plt


OBS_GRID_HALF_SIZE = 5
OBS_GRID_SIZE = OBS_GRID_HALF_SIZE * 2 + 1
OBS_GRID_CHANNELS = 4

OBS_GRID_SHAPE = (OBS_GRID_CHANNELS, OBS_GRID_SIZE, OBS_GRID_SIZE)

OBS_VECTOR_LEN = 5


def expand_grid(grid):
    expanded_grid_size = (
        grid.shape[0],
        grid.shape[1] + 2 * OBS_GRID_HALF_SIZE,
        grid.shape[2] + 2 * OBS_GRID_HALF_SIZE,
    )
    expanded_grid = np.zeros(expanded_grid_size)
    expanded_grid[-1] = 1

    expanded_grid[
        :,
        OBS_GRID_HALF_SIZE:-OBS_GRID_HALF_SIZE,
        OBS_GRID_HALF_SIZE:-OBS_GRID_HALF_SIZE,
    ] = grid
    return expanded_grid


def get_local_grid(grid, pos):
    return grid[
        :,
        pos[1] : pos[1] + 2 * OBS_GRID_HALF_SIZE + 1,
        pos[0] : pos[0] + 2 * OBS_GRID_HALF_SIZE + 1,
    ]


def closest_delta(all_pos, pos):
    closest_id = np.argmin(np.sum(np.abs(all_pos - pos), axis=1))
    return all_pos[closest_id] - pos


def calc_unit_obs(obs):

    # pre_computation of the full grid features
    full_grid = np.zeros((5,) + obs["player_0"]["board"]["ice"].shape)
    full_grid[0] = obs["player_0"]["board"]["ice"]
    full_grid[1] = obs["player_0"]["board"]["ore"]

    for team in teams:
        for unit in obs[team]["units"][team].values():
            full_grid[2, unit["pos"][0], unit["pos"][1]] = 1
    for team in teams:
        for factory in obs[team]["factories"][team].values():
            full_grid[3, factory["pos"][0], factory["pos"][1]] = 1

    expanded_grid = expand_grid(full_grid)

    # pre_computation of the vector features
    factories_pos = {team: [] for team in teams}
    for team in teams:
        for factory in obs[team]["factories"][team].values():
            factories_pos[team].append(factory["pos"])
    factories_pos = {key: np.array(value) for key, value in factories_pos.items()}

    to_return = {team: {} for team in teams}
    for team in teams:
        for unit_name, unit in obs[team]["units"][team].items():

            # filling the grid
            grid = get_local_grid(expanded_grid, unit["pos"])

            # filling the vector
            vector = np.zeros(OBS_VECTOR_LEN)
            vector[0:2] = closest_delta(factories_pos[team], unit["pos"]) / 5
            vector[2] = unit["cargo"]["ice"] / 100
            vector[3] = unit["cargo"]["ore"] / 100
            vector[4] = unit["power"] / 150

            to_return[team][unit_name] = {"grid": grid, "vector": vector}

    return to_return
