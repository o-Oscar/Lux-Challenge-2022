import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions.categorical import Categorical

from utils.agent.base import BaseAgent
from utils.obs.base import BaseObsGenerator
from utils.action.base import BaseActionHandler
from utils import teams

BIG_NUMBER = 1e10


class NullAgent(BaseAgent):
    def __init__(self, heavy_robot: bool = True):
        super().__init__()
        if heavy_robot:
            self.cargo_space = 1000
        else:
            self.cargo_space = 100

    def get_action(self, obs, masks, action=None):

        logits = th.zeros([2, 7, obs.shape[2], obs.shape[3]])

        # logits = self.actor(obs)
        logits = logits * masks - BIG_NUMBER * (1 - masks)
        logits = logits.permute(0, 2, 3, 1)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        # action = th.argmax(logits, dim=-1) * 0

        return (action, probs.log_prob(action), probs.entropy(), th.zeros([2, 48, 48]))
