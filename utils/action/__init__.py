from enum import Enum
from base import BaseActionHandler
from move import MoveActionHandler
from harvest import HarvestActionHandler


class ActionHandler(Enum):
    BASE = BaseActionHandler()
    MOVE = MoveActionHandler()
    HARVEST = HarvestActionHandler()