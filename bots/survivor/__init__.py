from utils.action.move import MoveActionHandler
from utils.obs.position import PositionObsGenerator
from utils.reward.survivor_move import SurvivorMoveRewardGenerator

ACTION_HANDLER = MoveActionHandler()
OBS_GENERATOR = PositionObsGenerator()
REWARD_GENERATOR = SurvivorMoveRewardGenerator()
