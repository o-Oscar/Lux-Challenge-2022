import numpy as np

from utils import teams
from utils.reward.base import BaseRewardGenerator

DEATH_REWARD = -1
ON_WATER_REWARD = 0.1


def expand(grid: np.ndarray, border: int, default: np.float32 = 0.0) -> np.ndarray:
    return_shape = [x + 2 * border for x in grid.shape]
    to_return = np.zeros(return_shape) + default
    to_return[border:-border, border:-border] = grid
    return to_return


def contract(grid: np.ndarray, border: int):
    return grid[border:-border, border:-border]


class ThirstyReward(BaseRewardGenerator):
    def __init__(self):
        pass

    def calc_rewards(self, old_obs, actions, obs):

        to_return = {}

        kernel_half_size = 1
        kernel_size = kernel_half_size * 2 + 1
        kernel = np.ones((kernel_size, kernel_size))

        for team in teams:
            factory_grid = np.zeros(obs["player_0"]["board"]["ice"].shape)
            for factory in obs[team]["factories"][team].values():
                factory_grid[factory["pos"][0], factory["pos"][1]] = 1

            units_grid = np.zeros(obs["player_0"]["board"]["ice"].shape)
            for old_unit in old_obs[team]["units"][team].values():
                units_grid[old_unit["pos"][0], old_unit["pos"][1]] = 1
            units_grid = expand(units_grid, kernel_half_size)

            ice_grid = obs[team]["board"]["ice"]

            reward_grid = np.zeros(obs["player_0"]["board"]["ice"].shape)
            reward_grid = expand(reward_grid, kernel_half_size)

            for unit_id, old_unit in old_obs[team]["units"][team].items():

                cur_reward = 0

                # has the unit died
                if unit_id not in obs[team]["units"][team]:
                    cur_reward += DEATH_REWARD

                # is the unit in the middle of a factory (where other units could spawn)
                if unit_id in obs[team]["units"][team]:
                    unit = obs[team]["units"][team][unit_id]
                    if ice_grid[unit["pos"][0], unit["pos"][1]] == 1:
                        cur_reward += ON_WATER_REWARD

                pos = old_unit["pos"]
                n_units = np.sum(
                    units_grid[
                        pos[0] : pos[0] + kernel_size,
                        pos[1] : pos[1] + kernel_size,
                    ]
                )
                reward_grid[
                    pos[0] : pos[0] + kernel_size,
                    pos[1] : pos[1] + kernel_size,
                ] += (
                    cur_reward / n_units
                )

            to_return[team] = contract(reward_grid, kernel_half_size)

        return to_return
