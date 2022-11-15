from enum import Enum

from base import BaseRewardGenerator
from survivor import SurvivorRewardGenerator
from survivor_move import SurvivorMoveRewardGenerator

class Reward(Enum):
    BASE = BaseRewardGenerator()
    SURVIVOR = SurvivorRewardGenerator()
    SURVIVOR_MOVE = SurvivorMoveRewardGenerator()