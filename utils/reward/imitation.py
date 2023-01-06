import numpy as np

from utils import teams
from utils.reward.base import BaseRewardGenerator

from utils.agent.baseline import BaselineAgent

BASE_REWARD = 1


class ImitationRewardGenerator(BaseRewardGenerator):
    def __init__(self, heavy_robot: bool = True):
        if heavy_robot:
            self.cargo_space = 1000
        else:
            self.cargo_space = 100
        self.baseline_agent = BaselineAgent()
        pass

    def calc_rewards(self, old_obs, actions, obs):

        to_return = {}
        to_return_monitoring = {}

        imitation_action_grid = self.baseline_agent.get_luxai_action(old_obs)

        for team in teams:

            # Initialisation of the reward grid
            reward_grid = np.zeros(obs["player_0"]["board"]["ice"].shape)
            reward_grid_monitoring = np.zeros(obs["player_0"]["board"]["ice"].shape)

            for unit_id, old_unit in old_obs[team]["units"][team].items():
                cur_reward = 0
                cur_reward_monitoring = 0
                # Is the unit still alive ?
                if unit_id in obs[team]["units"][team]:

                    unit = obs[team]["units"][team][unit_id]

                    # What did the reward agent ?
                    baseline_action = imitation_action_grid[
                        old_unit["pos"][0], old_unit["pos"][1]
                    ]
                    if (
                        unit_id in actions[team]
                        and (baseline_action == actions[team][unit_id][0][:2]).all()
                    ):
                        cur_reward += BASE_REWARD

                    # Ice cargo
                    old_ice_cargo = old_obs[team]["units"][team][unit_id]["cargo"][
                        "ice"
                    ]
                    curr_ice_cargo = unit["cargo"]["ice"]

                    # Did it transfered ice to a factory ?
                    if (
                        curr_ice_cargo < old_ice_cargo
                        and unit_id in actions[team]
                        and (
                            actions[team][unit_id]
                            == np.array([[1, 0, 0, self.cargo_space, 0]])
                        ).all()
                    ):
                        cur_reward_monitoring += 1

                reward_grid[old_unit["pos"][0], old_unit["pos"][1]] = cur_reward
                reward_grid_monitoring[
                    old_unit["pos"][0], old_unit["pos"][1]
                ] = cur_reward_monitoring

            to_return[team] = reward_grid
            to_return_monitoring[team] = reward_grid_monitoring

        return to_return, to_return_monitoring
