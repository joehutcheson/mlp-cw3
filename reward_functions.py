

class RewardFunction:

    def __init__(self, env_, agent_):

        self.env = env_
        self.agent = agent_

    def get_reward(self):
        raise NotImplementedError()

