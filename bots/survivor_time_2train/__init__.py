from utils.action.move import MoveActionHandler
from utils.obs.position_time import PositionTimeObsGenerator
from utils.reward.survivor_move import SurvivorMoveRewardGenerator
from utils.reward.survivor_move_time import SurvivorMoveTimeRewardGenerator

ACTION_HANDLER = MoveActionHandler()
OBS_GENERATOR = PositionTimeObsGenerator()
REWARD_GENERATOR_1 = SurvivorMoveRewardGenerator()
REWARD_GENERATOR_2 = SurvivorMoveTimeRewardGenerator()
