from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN:

    def __init__(self):
        """
        Constructor
        """
        pass



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    """
    We’ll be using experience replay memory for training our DQN.
    It stores the transitions that the agent observes, allowing us to reuse this
    data later. By sampling from it randomly, the transitions that build up a
    batch are decorrelated. It has been shown that this greatly stabilizes and
    improves the DQN training procedure.
    """

    def __init__(self):
        """
        Constructor
        """
        pass
