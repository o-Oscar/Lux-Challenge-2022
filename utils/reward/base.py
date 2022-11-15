class BaseRewardGenerator:
    def __init__(self):
        pass


    def calc_rewards(self, old_obs, actions, obs):
        raise NotImplementedError
