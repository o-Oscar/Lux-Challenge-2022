import numpy as np
from utils import teams
from utils.reward.base import BaseRewardGenerator

GATHER_REWARD = 1


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

                # is the unit still alive
                if unit_id in obs[team]["units"][team]:
                    unit = obs[team]["units"][team][unit_id]
                    # is the unit gathering ice and does it have cargo available
                    if (
                        ice[unit["pos"][0], unit["pos"][1]] != 0
                        and unit_id in actions[team]
                        and (
                            actions[team][unit_id] == np.array([[3, 0, 0, 0, 0]])
                        ).all()
                    ):
                        cur_reward += GATHER_REWARD

                reward_grid[old_unit["pos"][0], old_unit["pos"][1]] = cur_reward

            to_return[team] = reward_grid

        return to_return
