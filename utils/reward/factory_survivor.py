import numpy as np
from utils import teams
from utils.reward.base import BaseRewardGenerator

DIST_TO_FACTORY_REWARD = -0.1
GATHER_REWARD = 1
FULL_REWARD = 50
TRANSFER_REWARD = 1000


class FactorySurvivorRewardGenerator(BaseRewardGenerator):
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
            factory_grid = np.zeros(obs["player_0"]["board"]["ice"].shape)

            for factory in obs[team]["factories"][team].values():
                factory_grid[factory["pos"][0], factory["pos"][1]] = 1

            all_x = np.arange(obs["player_0"]["board"]["ice"].shape[0])
            all_y = np.arange(obs["player_0"]["board"]["ice"].shape[1])
            all_x, all_y = np.meshgrid(all_x, all_y)
            distance_to_factories = np.zeros(obs["player_0"]["board"]["ice"].shape)
            all_deltas = []
            for factory in obs[team]["factories"][team].values():
                cur_delta = np.abs(all_x - factory["pos"][0]) + np.abs(
                    all_y - factory["pos"][1]
                )
                all_deltas.append(cur_delta)
            if len(all_deltas) > 0:
                distance_to_factories = np.min(all_deltas, axis=0)

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
                        # is the robot just filled ?
                        if old_ice_cargo < curr_ice_cargo:
                            cur_reward += FULL_REWARD
                        # distance to the closest factory
                        cur_reward += (
                            DIST_TO_FACTORY_REWARD
                            * distance_to_factories[unit["pos"][0], unit["pos"][1]]
                        )

                    # did he transfered ice to a factory ?
                    if (
                        curr_ice_cargo < old_ice_cargo
                        and unit_id in actions[team]
                        and (
                            actions[team][unit_id]
                            == np.array([[1, 0, 0, self.cargo_space, 0]])
                        ).all()
                    ):
                        cur_reward += TRANSFER_REWARD
                        # print()
                        # print()
                        # print("#" * len("WE DID IT !"))
                        # print(
                        #     "WE DID IT ! ("
                        #     + str(obs[team]["real_env_steps"])
                        #     + "th step)"
                        # )
                        # print("#" * len("WE DID IT !"))
                        # print()
                        # print()

                reward_grid[old_unit["pos"][0], old_unit["pos"][1]] = cur_reward

            to_return[team] = reward_grid

        return to_return
