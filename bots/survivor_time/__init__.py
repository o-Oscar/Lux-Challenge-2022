from utils.action.harvest import HarvestActionHandler
from utils.action.move import MoveActionHandler
from utils.obs.complete import CompleteObsGenerator
from utils.reward.survivor import SurvivorRewardGenerator
from utils.reward.survivor_move_time import SurvivorMoveRewardGenerator

ACTION_HANDLER = MoveActionHandler()
OBS_GENERATOR = CompleteObsGenerator()
REWARD_GENERATOR = SurvivorMoveRewardGenerator()
