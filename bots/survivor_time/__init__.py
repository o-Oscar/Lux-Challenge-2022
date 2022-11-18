from utils.action.move import MoveActionHandler
from utils.obs.position_time import PositionTimeObsGenerator
from utils.reward.survivor_move_time import SurvivorMoveTimeRewardGenerator

ACTION_HANDLER = MoveActionHandler()
OBS_GENERATOR = PositionTimeObsGenerator()
REWARD_GENERATOR = SurvivorMoveTimeRewardGenerator()
