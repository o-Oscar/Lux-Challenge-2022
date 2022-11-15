from enum import Enum

from base import BaseObsGenerator
from complete import CompleteObsGenerator
from minimal import MinimalObsGenerator

class Obs(Enum):
    BASE = BaseObsGenerator()
    COMPLETE = CompleteObsGenerator()
    MINIMAL = MinimalObsGenerator()