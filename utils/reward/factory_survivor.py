import numpy as np
from utils import teams
from utils.reward.base import BaseRewardGenerator

DEATH_REWARD = -1
ON_SPAWN_REWARD = -0.1
DIST_TO_FACTORY_REWARD = -0.1
FULL_REWARD = 3
GATHER_REWARD = 0.1
TRANSFER_REWARD = 1


class FactorySurvivorRewardGenerator(BaseRewardGenerator):
    def __init__(self):
        pass

    def calc_rewards(self, old_obs, actions, obs):

        to_return = {}

        ice = obs["player_0"]["board"]["ice"]

        for team in teams:
            factory_grid = np.zeros(obs["player_0"]["board"]["ice"].shape)

            for factory in obs[team]["factories"][team].values():
                factory_grid[factory["pos"][1], factory["pos"][0]] = 1

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

                # has the unit died
                if unit_id not in obs[team]["units"][team]:
                    cur_reward += DEATH_REWARD

                if unit_id in obs[team]["units"][team]:
                    unit = obs[team]["units"][team][unit_id]
                    # is the unit in the middle of a factory (where other units could spawn)
                    if factory_grid[unit["pos"][1], unit["pos"][0]] == 1:
                        cur_reward += ON_SPAWN_REWARD

                    # is the unit gathering ice and does it have cargo available
                    if (
                        ice[unit["pos"][1], unit["pos"][0]] != 0
                        and unit["cargo"]["ice"] < 100
                        and unit_id in actions[team]
                        and (
                            actions[team][unit_id] == np.array([[3, 0, 0, 0, 0]])
                        ).all()
                    ):
                        cur_reward += GATHER_REWARD

                    # is the unit full ? How far is it to the closest factory ?
                    if unit["cargo"]["ice"] == 100:
                        cur_reward += (
                            FULL_REWARD
                            + DIST_TO_FACTORY_REWARD
                            * distance_to_factories[unit["pos"][1], unit["pos"][0]]
                        )

                    # did the unit transfer ice to a factory
                    if unit_id in old_obs[team]["units"][team]:
                        old_ice_cargo = old_obs[team]["units"][team][unit_id]["cargo"][
                            "ice"
                        ]
                        curr_ice_cargo = unit["cargo"]["ice"]
                        if curr_ice_cargo < old_ice_cargo:
                            cur_reward += TRANSFER_REWARD * (
                                old_ice_cargo - curr_ice_cargo
                            )
                            print()
                            print()
                            print("#" * len("WE DID IT !"))
                            print("WE DID IT !")
                            print("#" * len("WE DID IT !"))
                            print()
                            print()

                reward_grid[old_unit["pos"][1], old_unit["pos"][0]] = cur_reward

            to_return[team] = reward_grid

        return to_return
