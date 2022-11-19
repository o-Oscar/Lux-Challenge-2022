from utils.action.base import BaseActionHandler
from utils.action.move import MoveActionHandler
from utils.action.harvest import HarvestActionHandler

from utils.agent.base import BaseAgent
from utils.agent.conv import ConvAgent

from utils.obs.base import BaseObsGenerator
from utils.obs.position import PositionObsGenerator
from utils.obs.position_time import PositionTimeObsGenerator

from utils.reward.base import BaseRewardGenerator
from utils.reward.survivor_move import SurvivorMoveRewardGenerator
from utils.reward.survivor_dance import SurvivorDanceRewardGenerator


Bot_List = ["survivor", "dance", "big_dance", "dance_2train"]


class Bot:
    action: BaseActionHandler
    agent: BaseAgent
    obs_generator: BaseObsGenerator
    reward_generators: list
    reward_update_nbs: list  # limits of step for each reward (sequential learning)

    def __init__(self, bot_type: str):
        if bot_type not in Bot_List:
            print("Bot not created, he must be in", Bot_List)
            raise ValueError

        if bot_type == "survivor":
            self.action = MoveActionHandler()
            self.obs_generator = PositionObsGenerator()
            self.agent = ConvAgent(self.obs_generator, self.action)
            self.reward_generators = [SurvivorMoveRewardGenerator()]
            self.reward_update_nbs = [1000]

        if bot_type == "dance":
            self.action = MoveActionHandler()
            self.obs_generator = PositionTimeObsGenerator()
            self.agent = ConvAgent(self.obs_generator, self.action)
            self.reward_generators = [SurvivorDanceRewardGenerator()]
            self.reward_update_nbs = [1000]

        if bot_type == "big_dance":
            self.action = MoveActionHandler()
            self.obs_generator = PositionTimeObsGenerator()
            self.agent = (
                ConvAgent(
                    self.obs_generator,
                    self.action,
                    inside_dim=128,
                    grid_kernel_size=22,
                    grid_layers_nb=2,
                    post_obs_layers_nb=2,
                ),
            )
            self.reward_generators = [SurvivorDanceRewardGenerator()]
            self.reward_update_nbs = [1000]

        if bot_type == "dance_2train":
            self.action = MoveActionHandler()
            self.obs_generator = PositionTimeObsGenerator()
            self.agent = ConvAgent(self.obs_generator, self.action)
            self.reward_generators = [
                SurvivorMoveRewardGenerator(),
                SurvivorDanceRewardGenerator(),
            ]
            self.reward_update_nbs = [100, 1000]
