from utils.action.move import MoveActionHandler
from utils.obs.minimal import MinimalObsGenerator
from utils.reward.survivor import SurvivorRewardGenerator

ACTION_HANDLER = MoveActionHandler()
OBS_GENERATOR = MinimalObsGenerator()
REWARD = SurvivorRewardGenerator()