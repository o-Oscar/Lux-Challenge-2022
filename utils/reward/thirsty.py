import numpy as np
from utils import teams
from utils.reward.base import BaseRewardGenerator

DEATH_REWARD = -1
ON_SPAWN_REWARD = -0.1
GATHER_REWARD = 0.01


class ThirstyRewardGenerator(BaseRewardGenerator):
    def __init__(self):
        pass

    def calc_rewards(self, old_obs, actions, obs):

        to_return = {}

        ice_ore_grid = obs["player_0"]["board"]["ice"] + obs["player_0"]["board"]["ore"]

        for team in teams:
            factory_grid = np.zeros(obs["player_0"]["board"]["ice"].shape)

            for factory in obs[team]["factories"][team].values():
                factory_grid[factory["pos"][1], factory["pos"][0]] = 1

            reward_grid = np.zeros(obs["player_0"]["board"]["ice"].shape)

            for unit_id, old_unit in old_obs[team]["units"][team].items():

                cur_reward = 0

                # has the unit died
                if unit_id not in obs[team]["units"][team]:
                    cur_reward += DEATH_REWARD

                if unit_id in obs[team]["units"][team]:
                    unit = obs[team]["units"][team][unit_id]
                    # is the unit in the middle of a factory (where other units could spawn)
                    if factory_grid[unit["pos"][1], unit["pos"][0]] == 1:
                        cur_reward += ON_SPAWN_REWARD

                    # is the unit on an ice/ore case

                    if (
                        ice_ore_grid[unit["pos"][1], unit["pos"][0]] != 0
                        and unit_id in actions[team]
                        and (
                            actions[team][unit_id] == np.array([[3, 0, 0, 0, 0]])
                        ).all()
                    ):
                        cur_reward += GATHER_REWARD

                reward_grid[old_unit["pos"][1], old_unit["pos"][0]] = cur_reward

            to_return[team] = reward_grid

        return to_return
