class BaseObsGenerator:
    def __init__(self):
        self.channel_nb: int
        self.grid_channel_nb: int
        self.vector_channel_nb: int
        pass

    def calc_obs(self, obs):
        raise NotImplementedError
