import numpy as np
from utils import teams
from utils.reward.base import BaseRewardGenerator

DEATH_REWARD = -1
GATHER_REWARD = 1
FULL_REWARD = 5


class ThirstyRewardGenerator(BaseRewardGenerator):
    def __init__(self, heavy_robot: bool = True):
        if heavy_robot:
            self.cargo_space = 1000
        else:
            self.cargo_space = 100
        pass

    def calc_rewards(self, old_obs, actions, obs):

        to_return = {}

        ice = obs["player_0"]["board"]["ice"]

        for team in teams:
            reward_grid = np.zeros(obs["player_0"]["board"]["ice"].shape)

            for unit_id, old_unit in old_obs[team]["units"][team].items():

                cur_reward = 0

                # has the unit died
                if unit_id not in obs[team]["units"][team]:
                    cur_reward += DEATH_REWARD

                else:
                    unit = obs[team]["units"][team][unit_id]
                    # is the unit gathering ice and does it have cargo available
                    if (
                        ice[unit["pos"][1], unit["pos"][0]] != 0
                        and unit_id in actions[team]
                        and (
                            actions[team][unit_id] == np.array([[3, 0, 0, 0, 0]])
                        ).all()
                    ):
                        if unit["cargo"]["ice"] < self.cargo_space:
                            cur_reward += GATHER_REWARD
                        else:
                            cur_reward -= GATHER_REWARD

                    # ice cargo
                    old_ice_cargo = old_obs[team]["units"][team][unit_id]["cargo"][
                        "ice"
                    ]
                    curr_ice_cargo = unit["cargo"]["ice"]

                    if curr_ice_cargo == self.cargo_space:
                        if old_ice_cargo < curr_ice_cargo:
                            cur_reward += FULL_REWARD

                reward_grid[old_unit["pos"][1], old_unit["pos"][0]] = cur_reward

            to_return[team] = reward_grid

        return to_return
