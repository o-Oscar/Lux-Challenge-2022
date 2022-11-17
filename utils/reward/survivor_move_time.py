import numpy as np
from utils import teams
from utils.reward.base import BaseRewardGenerator


DEATH_REWARD = -1
ON_SPAWN_REWARD = -0.1
MOVE_REWARD = 0.01


class SurvivorMoveRewardGenerator(BaseRewardGenerator):
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

                if obs[team]["real_env_steps"] % 50 > 25:
                    direction = "right"
                else:
                    direction = "left"

                if unit_id in actions[team]:
                    if (
                        direction == "right"
                        and (
                            actions[team][unit_id] == np.array([[0, 2, 0, 0, 0]])
                        ).all()
                    ) or (
                        direction == "left"
                        and (
                            actions[team][unit_id] == np.array([[0, 4, 0, 0, 0]])
                        ).all()
                    ):
                        cur_reward += MOVE_REWARD

                reward_grid[old_unit["pos"][1], old_unit["pos"][0]] = cur_reward

            to_return[team] = reward_grid

        return to_return
