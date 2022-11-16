from utils.action.harvest import HarvestActionHandler
from utils.action.move import MoveActionHandler
from utils.obs.minimal import MinimalObsGenerator
from utils.reward.survivor import SurvivorRewardGenerator
from utils.reward.survivor_move import SurvivorMoveRewardGenerator

ACTION_HANDLER = MoveActionHandler()
OBS_GENERATOR = MinimalObsGenerator()
REWARD_GENERATOR = SurvivorMoveRewardGenerator()
