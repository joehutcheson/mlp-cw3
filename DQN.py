import random
from collections import namedtuple, deque

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
    Replay memory is a data structure that stores a history of past experiences
    or interactions in the environment. These experiences are represented as
    tuples of (state, action, reward, next state) and are randomly sampled from
    the replay memory during the learning process.
    """

    def __init__(self, capacity):
        """ Constructor """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """ Sample an experience from the replay memory of size batch_size"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """ Pretty straight forward - returns the size of the replay memory"""
        return len(self.memory)
