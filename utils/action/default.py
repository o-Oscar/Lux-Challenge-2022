from utils.action import ActionHandler
from utils import teams
import numpy as np


class DefaultActionHandler(ActionHandler):
    def __init__(self):
        super().__init__()

        self.robot_to_env_actions = [
            None,  # ne rien faire
            np.array([0, 1, 0, 0, 0]),  # bouger en haut
            np.array([0, 2, 0, 0, 0]),  # bouger à droite
            np.array([0, 3, 0, 0, 0]),  # bouger en bas
            np.array([0, 4, 0, 0, 0]),  # bouger à gauche
            np.array([1, 0, 0, 100, 0]),  # transférer de la glace
            np.array([1, 0, 1, 100, 0]),  # transférer des minerais
            np.array([2, 0, 4, 50, 0]),  # transférer des minerais
            np.array([2, 0, 4, 100, 0]),  # transférer des minerais
            np.array([2, 0, 4, 150, 0]),  # transférer des minerais
            np.array([3, 0, 0, 0, 0]),  # creuser
        ]

        self.action_nb = len(self.robot_to_env_actions)

    def calc_masks(self, obs):

        to_return = {team: {} for team in teams}

        factory_mask = np.zeros(obs["player_0"]["board"]["ice"].shape)
        for team in teams:
            for factory in obs[team]["factories"][team].values():
                factory_mask[
                    factory["pos"][1] - 1 : factory["pos"][1] + 2,
                    factory["pos"][0] - 1 : factory["pos"][0] + 2,
                ] = 1

        for team in teams:
            for unit_name, unit in obs[team]["units"][team].items():

                mask = np.ones(11)

                # up action not possible when on the top of the board
                if unit["pos"][1] == 0:
                    mask[1] = 0
                # right action not possible when on the right of the board
                if unit["pos"][0] == obs[team]["board"]["ice"].shape[0] - 1:
                    mask[2] = 0
                # down action not possible when on the bottom of the board
                if unit["pos"][1] == obs[team]["board"]["ice"].shape[1] - 1:
                    mask[3] = 0
                # left action not possible when on the left of the board
                if unit["pos"][0] == 0:
                    mask[4] = 0
                # ice transfer not possible when we have no ice or we are not on top of a factory
                if (
                    unit["cargo"]["ice"] <= 0
                    or factory_mask[unit["pos"][1], unit["pos"][0]] == 0
                ):
                    mask[5] = 0
                # ore transfer not possible when we have no ice or we are not on top of a factory
                if (
                    unit["cargo"]["ore"] <= 0
                    or factory_mask[unit["pos"][1], unit["pos"][0]] == 0
                ):
                    mask[6] = 0
                # energy transfer not possible when not on top of a factory
                if factory_mask[unit["pos"][1], unit["pos"][0]] == 0:
                    mask[7] = 0
                    mask[8] = 0
                    mask[9] = 0
                # digging not possible when on top of factory or not on top of a square with rubble, ice or ore
                if (
                    obs[team]["board"]["ice"][unit["pos"][1], unit["pos"][0]] == 0
                    and obs[team]["board"]["ore"][unit["pos"][1], unit["pos"][0]] == 0
                    and obs[team]["board"]["rubble"][unit["pos"][1], unit["pos"][0]]
                    == 0
                ):
                    mask[10] = 0
                if factory_mask[unit["pos"][1], unit["pos"][0]] == 1:
                    mask[7] = 1

                to_return[team][unit_name] = mask

        return to_return

    def network_to_robots(self, network_actions):
        to_return = {team: {} for team in teams}
        for team in teams:
            for unit_name, unit_action in network_actions[team].items():
                cur_action = self.robot_to_env_actions[unit_action]
                if cur_action is not None:
                    to_return[team][unit_name] = cur_action
        return to_return

    def robots_to_network(self, robot_actions):
        raise NotImplementedError
