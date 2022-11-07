import numpy as np
from utils import teams

DEATH_REWARD = -1
ON_SPAWN_REWARD = -0.1


class DefaultRewardGenerator:
    def __init__(self):
        pass

    def calc_rewards(self, old_obs, actions, obs):

        to_return = {team: {} for team in teams}

        factory_grid = np.zeros(obs["player_0"]["board"]["ice"].shape)

        for team in teams:
            for factory in obs[team]["factories"][team].values():
                factory_grid[factory["pos"][1], factory["pos"][0]] = 1

        for team in teams:
            for unit_id, old_unit in old_obs[team]["units"][team].items():

                cur_reward = 0

                # has the unit died
                if unit_id not in obs[team]["units"][team]:
                    cur_reward += DEATH_REWARD

                # is the unit in the middle of a factory (where other units could spawn)
                if unit_id in obs[team]["units"][team]:
                    unit = obs[team]["units"][team][unit_id]
                    if factory_grid[unit["pos"][1], unit["pos"][0]] == 1:
                        cur_reward += ON_SPAWN_REWARD

                # # the unit has gone to the right
                # cur_reward = 0
                # if unit_id in actions[team]:
                #     if np.array_equal(
                #         actions[team][unit_id], np.array([0, 1, 0, 0, 0])
                #     ):
                #         cur_reward += 1

                to_return[team][unit_id] = cur_reward

        return to_return
