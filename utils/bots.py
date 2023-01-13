from utils.action.base import BaseActionHandler
from utils.action.move import MoveActionHandler
from utils.action.harvest import HarvestActionHandler
from utils.action.harvest_transfer import HarvestTransferActionHandler

from utils.agent.base import BaseAgent
from utils.agent.conv import ConvAgent

from utils.obs.base import BaseObsGenerator
from utils.obs.position import PositionObsGenerator
from utils.obs.position_time import PositionTimeObsGenerator
from utils.obs.position_ice_ore import PositionIceOreObsGenerator
from utils.obs.position_ice_factory import PositionIceFactoryObsGenerator

from utils.reward.base import BaseRewardGenerator
from utils.reward.survivor_move import SurvivorMoveRewardGenerator
from utils.reward.survivor_dance import SurvivorDanceRewardGenerator
from utils.reward.thirsty import ThirstyRewardGenerator
from utils.reward.factory_survivor import FactorySurvivorRewardGenerator
from utils.reward.imitation import ImitationRewardGenerator


class Bot:
    action: BaseActionHandler
    agent: BaseAgent
    obs_generator: BaseObsGenerator
    reward_generators: list
    reward_update_nbs: list  # limits of step for each reward (sequential learning)

    def __init__(self, bot_type: str, vec_chan: int, use_relu: bool = False):

        # TEST
        if bot_type == "test":
            self.action = HarvestTransferActionHandler()
            self.obs_generator = PositionIceFactoryObsGenerator()
            self.agent = ConvAgent(
                self.obs_generator,
                self.action,
                grid_kernel_size=1,
                inside_layers_nb=0,
                final_kernel_size=1,
                final_layers_nb=1,
                use_relu=use_relu,
            )
            self.reward_generators = [FactorySurvivorRewardGenerator()]
            self.reward_update_nbs = [5]

        # LIGHT BOTS
        elif "light" in bot_type:
            if "imitator" in bot_type:
                print()
                print(
                    "imitator detected, set reward_generator to ImitationRewardGenerator"
                )
                reward_generators = [ImitationRewardGenerator()]
            elif "factory_survivor" in bot_type:
                print(
                    "factory_survivor detected, set reward_generator to FactorySurvivorRewardGenerator"
                )
                reward_generators = [FactorySurvivorRewardGenerator()]
            else:
                raise NameError("No known reward generator")

            if "super_light" in bot_type:
                print()
                print("super_light detected, change grid_kernel_size to 11")
                grid_kernel_size = 11
            else:
                grid_kernel_size = 21
            if "light_deep" in bot_type:
                print()
                print("light_deep detected, change inside_layers_nb to 1")
                inside_layers_nb = 1
            else:
                inside_layers_nb = 0
            print()

            self.action = HarvestTransferActionHandler()
            self.obs_generator = PositionIceFactoryObsGenerator()
            self.agent = ConvAgent(
                self.obs_generator,
                self.action,
                grid_kernel_size=grid_kernel_size,
                vector_post_channel_nb=vec_chan,
                inside_layers_nb=inside_layers_nb,
                final_kernel_size=5,
                final_layers_nb=1,
                use_relu=use_relu,
            )
            self.reward_generators = reward_generators
            self.reward_update_nbs = [10000]

        ##################################################################################################
        ############## DEPRECIATED BECAUSE OF REWARD_GENERATOR NOT GIVING MONITORING REWARD ##############
        ##################################################################################################

        # THIRSTY
        elif bot_type == "thirsty_FK1_FL1":
            self.action = HarvestActionHandler()
            self.obs_generator = PositionIceOreObsGenerator()
            self.agent = ConvAgent(self.obs_generator, self.action)
            self.reward_generators = [ThirstyRewardGenerator()]
            self.reward_update_nbs = [1000]

        elif bot_type == "thirsty_FK5_FL1":
            self.action = HarvestActionHandler()
            self.obs_generator = PositionIceOreObsGenerator()
            self.agent = ConvAgent(self.obs_generator, self.action, final_kernel_size=5)
            self.reward_generators = [ThirstyRewardGenerator()]
            self.reward_update_nbs = [1000]

        elif bot_type == "thirsty_FK5_FL2":
            self.action = HarvestActionHandler()
            self.obs_generator = PositionIceOreObsGenerator()
            self.agent = ConvAgent(
                self.obs_generator, self.action, final_kernel_size=5, final_layers_nb=2
            )
            self.reward_generators = [ThirstyRewardGenerator()]
            self.reward_update_nbs = [1000]

        # DANCE
        elif bot_type == "dance_2train":
            self.action = MoveActionHandler()
            self.obs_generator = PositionTimeObsGenerator()
            self.agent = ConvAgent(self.obs_generator, self.action)
            self.reward_generators = [
                SurvivorMoveRewardGenerator(),
                SurvivorDanceRewardGenerator(),
            ]
            self.reward_update_nbs = [100, 1000]

        elif bot_type == "dance":
            self.action = MoveActionHandler()
            self.obs_generator = PositionTimeObsGenerator()
            self.agent = ConvAgent(self.obs_generator, self.action)
            self.reward_generators = [SurvivorDanceRewardGenerator()]
            self.reward_update_nbs = [1000]

        # SURVIVOR
        elif bot_type == "survivor":
            self.action = MoveActionHandler()
            self.obs_generator = PositionObsGenerator()
            self.agent = ConvAgent(self.obs_generator, self.action)
            self.reward_generators = [SurvivorMoveRewardGenerator()]
            self.reward_update_nbs = [1000]

        else:
            print("Bot not created")
            raise ValueError
