

import gym
from gym import spaces
import numpy as np


class Env (gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=np.zeros(self.observation_shape),
                                            high=np.ones(
                                                self.observation_shape),
                                            dtype=np.float16)

        self.action_space = spaces.Discrete(6,)
