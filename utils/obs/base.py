class BaseObsGenerator:
    def __init__(self):
        self.channel_nb = 0

    def calc_obs(self, obs):
        raise NotImplementedError
