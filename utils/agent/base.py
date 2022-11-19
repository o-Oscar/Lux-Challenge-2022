from utils.env import Env
import torch.nn as nn
import torch as th


class BaseAgent(nn.Module):
    def __init__(self):
        super().__init__()

    def get_value(self, obs):
        raise NotImplementedError

    def get_action(self, obs, masks, action=None):
        raise NotImplementedError
