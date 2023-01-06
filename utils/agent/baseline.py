import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions.categorical import Categorical

from utils.agent.base import BaseAgent
from utils.obs.base import BaseObsGenerator
from utils.action.base import BaseActionHandler
from utils import teams

BIG_NUMBER = 1e10


def get_closest(i_pos, j_pos, positions):
    if positions[i_pos, j_pos] == 1:
        return (i_pos, j_pos)
    for distance in range(positions.shape[0]):
        for i in range(distance):
            for delta_i in [-i, +i]:
                for delta_j in [-(distance - i), distance - i]:
                    target = (i_pos + delta_i, j_pos + delta_j)
                    if (
                        0 <= target[0] < positions.shape[0]
                        and 0 <= target[1] < positions.shape[1]
                        and positions[target[0]][target[1]] == 1
                    ):
                        return target
    return (0, 0)  # to not return None


class BaselineAgent(BaseAgent):
    def __init__(self, heavy_robot: bool = True):
        super().__init__()
        if heavy_robot:
            self.cargo_space = 1000
        else:
            self.cargo_space = 100

    def get_action(self, obs, masks, action=None):

        logits = th.zeros([2, 7, obs.shape[2], obs.shape[3]])

        for team in range(2):
            obs_team = obs[team]
            robots_positions = obs_team[0]
            factories_positions = obs_team[2]
            ice_positions = obs_team[4]
            fullyness = obs_team[5]

            for i in range(robots_positions.shape[0]):
                for j in range(robots_positions.shape[1]):
                    if robots_positions[i, j] == 1:
                        if fullyness[i, j] == 1:
                            (i_factory, j_factory) = get_closest(
                                i, j, factories_positions
                            )
                            if i_factory < i:
                                logits[team, 4, i, j] = BIG_NUMBER  # go west
                            elif i_factory > i:
                                logits[team, 2, i, j] = BIG_NUMBER  # go east
                            elif j_factory < j:
                                logits[team, 1, i, j] = BIG_NUMBER  # go north
                            elif j_factory > j:
                                logits[team, 3, i, j] = BIG_NUMBER  # go south
                            else:
                                logits[team, 5, i, j] = BIG_NUMBER  # transfer

                        else:
                            (i_ice, j_ice) = get_closest(i, j, ice_positions)
                            if i_ice < i:
                                logits[team, 4, i, j] = BIG_NUMBER  # go west
                            elif i_ice > i:
                                logits[team, 2, i, j] = BIG_NUMBER  # go east
                            elif j_ice < j:
                                logits[team, 1, i, j] = BIG_NUMBER  # go north
                            elif j_ice > j:
                                logits[team, 3, i, j] = BIG_NUMBER  # go south
                            else:
                                logits[team, 6, i, j] = BIG_NUMBER  # dig

        # logits = self.actor(obs)
        logits = logits * masks - BIG_NUMBER * (1 - masks)
        logits = logits.permute(0, 2, 3, 1)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        # action = th.argmax(logits, dim=-1) * 0

        return (action, probs.log_prob(action), probs.entropy(), th.zeros([2, 48, 48]))

    def get_luxai_action(self, obs):

        # carefull, here obs are luxai obs, not the one from the observator generator
        for team in teams:
            # We get back the good format for observation
            factories_positions = np.zeros_like(obs["player_0"]["board"]["ice"])
            for factory in obs[team]["factories"][team].values():
                factories_positions[
                    factory["pos"][0] - 1 : factory["pos"][0] + 2,
                    factory["pos"][1] - 1 : factory["pos"][1] + 2,
                ] = 1

            ice_positions = obs["player_0"]["board"]["ice"]

            actions = np.zeros(obs["player_0"]["board"]["ice"].shape + (2,))
            for unit_name, unit in obs[team]["units"][team].items():
                (i, j) = (unit["pos"][0], unit["pos"][1])
                if unit["cargo"]["ice"] == self.cargo_space:
                    (i_factory, j_factory) = get_closest(i, j, factories_positions)
                    if i_factory < i:
                        actions[i, j] = np.array([0, 4])  # go west
                    elif i_factory > i:
                        actions[i, j] = np.array([0, 2])  # go east
                    elif j_factory < j:
                        actions[i, j] = np.array([0, 1])  # go north
                    elif j_factory > j:
                        actions[i, j] = np.array([0, 3])  # go south
                    else:
                        actions[i, j] = np.array([1, 0])  # transfer

                else:
                    (i_ice, j_ice) = get_closest(i, j, ice_positions)
                    if i_ice < i:
                        actions[i, j] = np.array([0, 4])  # go west
                    elif i_ice > i:
                        actions[i, j] = np.array([0, 2])  # go east
                    elif j_ice < j:
                        actions[i, j] = np.array([0, 1])  # go north
                    elif j_ice > j:
                        actions[i, j] = np.array([0, 3])  # go south
                    else:
                        actions[i, j] = np.array([3, 0])  # dig
        return actions
