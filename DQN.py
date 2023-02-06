import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.module):
    """
    Our aim will be to train a policy that tries to maximize the discounted,
    cumulative reward
    """

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # Create a 3 layer netword with 128 hidden fc units per layer
        self.layer1 = nn.Linear(n_observations, 128)  # (in_f, out_f)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        # The feed forward method
        # Applying Rec Linear to each layer
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)  # Returns the final layer


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
