import numpy as np
from utils import teams
from utils.reward.base import BaseRewardGenerator

DEATH_REWARD = -1
ON_SPAWN_REWARD = -0.1


class SurvivorRewardGenerator(BaseRewardGenerator):
    def __init__(self):
        pass

    def calc_rewards(self, old_obs, actions, obs):

        to_return = {}

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

                # is the unit in the middle of a factory (where other units could spawn)
                # RQ : pour moi cette reward n'a pas lieu d'être : il doit apprendre tout seul qu'être devant une facto cest pas bien
                if unit_id in obs[team]["units"][team]:
                    unit = obs[team]["units"][team][unit_id]
                    if factory_grid[unit["pos"][1], unit["pos"][0]] == 1:
                        cur_reward += ON_SPAWN_REWARD

                reward_grid[old_unit["pos"][1], old_unit["pos"][0]] = cur_reward

            to_return[team] = reward_grid

        return to_return
