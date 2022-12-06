class BaseActionHandler:
    def __init__(self):
        self.action_nb = 0

    def calc_masks(self, obs):
        raise NotImplementedError

    def network_to_robots(self, network_actions):
        raise NotImplementedError

    def robots_to_network(self, robot_actions):
        raise NotImplementedError
