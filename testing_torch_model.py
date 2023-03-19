"""
ALL CODE HAS BEEN ADAPTED FROM
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pettingzoo.utils.wrappers.order_enforcing import OrderEnforcingWrapper

from stockfish import Stockfish

from checkmate import pettingzoo2stockfish, stockfish2pettingzoo
from reward_functions import normalize_rewards

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class Model:
    def __init__(self, env: OrderEnforcingWrapper,
                 reward_function, stockfish_path=None, reward_function_2=None, stockfish_difficulty=500,
                 use_same_model=False):

        """

        :param env: Petting Zoo chess environment
        :param reward_function: Reward function for first agent
        :param stockfish_path: Path to Stockfish binary
        :param reward_function_2: Reward function of second agent
        :param stockfish_difficulty: Stockfish difficulty level (0..20)
        :param use_same_model: Set true if agents to use the same model
        """

        # set up matplotlib
        plt.ion()

        # set to use either GPU or CPU or Apple Metal
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('GPU here')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print('MPS here')
        else:
            self.device = torch.device('cpu')
            print('CPU here')

        # PettingZoo setup
        self.env = env
        self.env.reset()

        # reward function for first agent
        self.reward_function = reward_function

        # Second agent setup
        self.stockfish = None
        self.reward_function_2 = None
        if stockfish_path is not None:
            # Second agent is stockfish
            self.stockfish = Stockfish(stockfish_path, depth=8, parameters={"Threads": 4, "Minimum Thinking Time": 0, "Ponder": True, "Hash": 1024, "Move Overhead": 0})
            self.stockfish.set_elo_rating(stockfish_difficulty)
        elif reward_function_2 is not None:
            # Second agent is our model
            self.reward_function_2 = reward_function_2
        else:
            raise Exception("Must define opponent model to be either Stockfish or a reward function.")

        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the AdamW optimizer
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4

        # Get number of actions from pettingzoo action space
        n_actions = env.action_space('player_0').n
        # Get the number of state observations
        # n_observations = math.prod(env.observe('player_0')['observation'].shape)
        # Model setup. Model is stores as dictionaries which are accessed by agent
        self.memory = dict()
        self.policy_net = dict()
        self.target_net = dict()
        self.optimizer = dict()
        for agent in self.env.agents:
            self.policy_net[agent] = Network().to(self.device)
            self.target_net[agent] = Network().to(self.device)
            self.target_net[agent].load_state_dict(self.policy_net[agent].state_dict())
            self.optimizer[agent] = optim.AdamW(self.policy_net[agent].parameters(), lr=self.LR, amsgrad=True)
            self.memory[agent] = ReplayMemory(10000)

        # Second model is set to point to the first model if using the same model for both agents
        # Each agent still has its own replay memory
        self.use_same_model = use_same_model
        if self.use_same_model:
            self.policy_net[self.env.agents[1]] = self.policy_net[self.env.agents[0]]
            self.target_net[self.env.agents[1]] = self.target_net[self.env.agents[0]]
            self.optimizer[self.env.agents[1]] = self.optimizer[self.env.agents[0]]

        self.steps_done = 0

        self.episode_durations = []

    def select_action(self, state, agent):
        # TODO: Have agents alternate taking first turn with each game
        assert self.stockfish or self.reward_function_2 is not None
        if self.stockfish and agent == self.env.agents[1]:
            # Stockfish decides move
            best_action = self.stockfish.get_best_move()
            return torch.tensor([[stockfish2pettingzoo(self.env, best_action)]], device=self.device, dtype=torch.long)
        else:
            # Our model decides move
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                            math.exp(-1. * self.steps_done / self.EPS_DECAY)
            self.steps_done += 1
            mask = self.env.observe(agent)['action_mask']
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return the largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    policy = self.policy_net[agent](state) * torch.from_numpy(mask).to(self.device)
                    if torch.max(policy) <= 0:
                        return torch.tensor([[self.env.action_space(agent).sample(mask)]], device=self.device,
                                            dtype=torch.long)
                    return policy.max(1)[1].view(1, 1)
            else:
                # Chooses a random move at the beginning of training
                return torch.tensor([[self.env.action_space(agent).sample(mask)]], device=self.device,
                                    dtype=torch.long)

    # def plot_durations(self, show_result=False):
    #     plt.figure(1)
    #     durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
    #     if show_result:
    #         plt.title('Result')
    #     else:
    #         plt.clf()
    #         plt.title('Training...')
    #     plt.xlabel('Episode')
    #     plt.ylabel('Duration')
    #     plt.plot(durations_t.numpy())
    #     # Take 100 episode averages and plot them too
    #     if len(durations_t) >= 100:
    #         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #         means = torch.cat((torch.zeros(99), means))
    #         plt.plot(means.numpy())
    #
    #     plt.pause(0.001)  # pause a bit so that plots are updated
    #     if is_ipython:
    #         if not show_result:
    #             display.display(plt.gcf())
    #             display.clear_output(wait=True)
    #         else:
    #             display.display(plt.gcf())

    def optimize_model(self, agent):
        if len(self.memory[agent]) < self.BATCH_SIZE:
            return
        transitions = self.memory[agent].sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been
        # taken for each batch state according to policy_net
        state_action_values = self.policy_net[agent](state_batch).gather(1,
                                                                         action_batch)

        # Compute V(s_{t+1}) for all next states. Expected values of actions
        # for non_final_next_states are computed based on the "older"
        # target_net; selecting their best reward with max(1)[0]. This is
        # merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = \
                self.target_net[agent](non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer[agent].zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net[agent].parameters(), 100)
        self.optimizer[agent].step()

    def train(self, num_episodes=500):

        for i_episode in range(num_episodes):

            # Initialize the environment and get its state
            self.env.reset()
            if self.stockfish:
                self.stockfish.set_position([])

            # Record the moves made in a game
            moves_made = []

            agent = self.env.agent_selection
            state = self.env.observe(agent)['observation']
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():

                # Get the next action
                action = self.select_action(state, agent)

                # Update the stockfish board if in use
                if self.stockfish:
                    move = pettingzoo2stockfish(self.env, action.item())
                    moves_made.append(move)
                    self.stockfish.make_moves_from_current_position([move])

                # Update the pettingzoo environment
                self.env.step(action.item())

                observation = self.env.observe(agent)['observation']

                # Calculate the reward
                if agent == self.env.agents[0]:
                    reward = self.reward_function(self.env) * 1000
                    reward = np.float32(normalize_rewards([reward, self.env.rewards['player_0']]))
                elif self.reward_function_2 is not None and agent == self.env.agents[1]:
                    reward = self.reward_function_2(self.env) * 1000
                    reward = np.float32(normalize_rewards([reward, self.env.rewards['player_1']]))
                elif self.stockfish is not None and agent == self.env.agents[1]:
                    reward = None
                else:
                    raise Exception

                # Check for end states
                terminated = self.env.terminations[agent]
                truncated = self.env.truncations[agent]
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32,
                                              device=self.device).unsqueeze(0)

                # Store the transition in memory
                if agent == self.env.agents[0] or self.reward_function_2 is not None:
                    reward = torch.tensor([reward], device=self.device)
                    self.memory[agent].push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if agent == 'player_0' or self.reward_function_2 is not None:
                    self.optimize_model(agent)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net[agent].state_dict()
                policy_net_state_dict = self.policy_net[agent].state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + \
                                                 target_net_state_dict[key] * (1 - self.TAU)
                self.target_net[agent].load_state_dict(target_net_state_dict)

                agent = self.env.agent_selection

                if done:
                    self.episode_durations.append(t + 1)
                    print(f"Game: {i_episode}, Result: {self.env.rewards}, Moves made: {t + 1}")
                    # with open("games.txt", "a") as myfile:
                    #     myfile.write(str(moves_made) + '\n')
                    # self.plot_durations()
                    break

        print('Complete')
        # self.plot_durations(show_result=True)
        # plt.ioff()
        # plt.show()

    def save_model(self, agent, path, name):
        torch.save(self.target_net[agent].state_dict(), path + name + '_target_net.pt')
        torch.save(self.policy_net[agent].state_dict(), path + name + '_policy_net.pt')

    def load_model(self, agent, path, name):
        self.target_net[agent].load_state_dict(torch.load(path + name + '_target_net.pt'))
        self.target_net[agent].eval()

        self.policy_net[agent].load_state_dict(torch.load(path + name + '_policy_net.pt'))
        self.policy_net[agent].eval()

    def test(self, num_episodes=25):

        results = {'player_0': 0, 'draw': 0, 'player_1': 0}

        for i_episode in range(num_episodes):

            # Initialize the environment and get its state
            self.env.reset()
            if self.stockfish:
                self.stockfish.set_position([])

            # Record the moves made in a game
            moves_made = []

            agent = self.env.agent_selection
            state = self.env.observe(agent)['observation']
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():

                # Get the next action
                action = self.select_action_test(state, agent)

                # Update the stockfish board if in use
                if self.stockfish:
                    move = pettingzoo2stockfish(self.env, action.item())
                    moves_made.append(move)
                    self.stockfish.make_moves_from_current_position([move])

                # Update the pettingzoo environment
                self.env.step(action.item())

                observation = self.env.observe(agent)['observation']

                # Check for end states
                terminated = self.env.terminations[agent]
                truncated = self.env.truncations[agent]
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32,
                                              device=self.device).unsqueeze(0)


                # Move to the next state
                state = next_state

                agent = self.env.agent_selection

                if done:
                    if self.env.rewards['player_0'] == 1:
                        results['player_0'] += 1
                    elif self.env.rewards['player_1'] == 1:
                        results['player_1'] += 1
                    else:
                        results['draw'] += 1
                    self.episode_durations.append(t + 1)
                    print(f"Game: {i_episode}, Result: {self.env.rewards}, Moves made: {t + 1}")
                    # with open("games.txt", "a") as myfile:
                    #     myfile.write(str(moves_made) + '\n')
                    # self.plot_durations()
                    break

        print(results)
        print('Complete')

    def select_action_test(self, state, agent):
        assert self.stockfish or self.reward_function_2 is not None
        if self.stockfish and agent == self.env.agents[1]:
            # Stockfish decides move
            best_action = self.stockfish.get_best_move()
            return torch.tensor([[stockfish2pettingzoo(self.env, best_action)]], device=self.device, dtype=torch.long)
        else:

            mask = self.env.observe(agent)['action_mask']
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                policy = self.policy_net[agent](state) * torch.from_numpy(mask).to(self.device)
                if torch.max(policy) <= 0:
                    return torch.tensor([[self.env.action_space(agent).sample(mask)]], device=self.device,
                                        dtype=torch.long)
                return policy.max(1)[1].view(1, 1)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 256)
        self.layer5 = nn.Linear(256, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)


# Taken from https://github.com/yukw777/leela-zero-pytorch/blob/master/leela_zero_pytorch/network.py
class ConvBlock(nn.Module):
    """
    A convolutional block with a convolution layer, batchnorm (with beta) and
    an optional relu
    Note on the bias for the convolutional layer:
    Leela Zero actually uses the bias for the convolutional layer to represent
    the learnable parameters (gamma and beta) of the following batch norm layer.
    This was done so that the format of the weights file, which only has one line
    for the layer weights and another for the bias, didn't have to change when
    batch norm layers were added.
    Currently, Leela Zero only uses the beta term of batch norm, and sets gamma to 1.
    Then, how do you actually use the convolutional bias to produce the same results
    as applying the learnable parameters in batch norm? Let's first take
    a look at the equation for batch norm:
    y = gamma * (x - mean)/sqrt(var - eps) + beta
    Since Leela Zero sets gamma to 1, the equation becomes:
    y = (x - mean)/sqrt(var - eps) + beta
    Now, let `x_conv` be the output of a convolutional layer without the bias.
    Then, we want to add some bias to `x_conv`, so that when you run it through
    batch norm without `beta`, the result is the same as running `x_conv`
    through the batch norm equation with only beta mentioned above. In an equation form:
    (x_conv + bias - mean)/sqrt(var - eps) = (x_conv - mean)/sqrt(var - eps) + beta
    x_conv + bias - mean = x_conv - mean + beta * sqrt(var - eps)
    bias = beta * sqrt(var - eps)
    So if we set the convolutional bias to `beta * sqrt(var - eps)`, we get the desired
    output, and this is what LeelaZero does.
    In Tensorflow, you can tell the batch norm layer to ignore just the gamma term
    by calling `tf.layers.batch_normalization(scale=False)` and be done with it.
    Unfortunately, in PyTorch you can't set batch normalization layers to ignore only
    `gamma`; you can only ignore both `gamma` and `beta` by setting the affine
    parameter to False: `BatchNorm2d(out_channels, affine=False)`. So, ConvBlock sets
    batch normalization to ignore both, then simply adds a tensor after, which
    represents `beta`.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, relu: bool = True
    ):
        super().__init__()
        # we only support the kernel sizes of 1 and 3
        assert kernel_size in (1, 3)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=1 if kernel_size == 3 else 0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.beta = nn.Parameter(torch.zeros(out_channels))  # type: ignore
        self.relu = relu

        # initializations
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x += self.beta.view(1, self.bn.num_features, 1, 1).expand_as(x)
        return F.relu(x, inplace=True) if self.relu else x


# Taken from https://github.com/yukw777/leela-zero-pytorch/blob/master/leela_zero_pytorch/network.py
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 3)
        self.conv2 = ConvBlock(out_channels, out_channels, 3, relu=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += identity
        return F.relu(out, inplace=True)


# Taken from https://github.com/yukw777/leela-zero-pytorch/blob/master/leela_zero_pytorch/network.py
class Network(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        board_size: int = 19,
        residual_channels: int = 128,
        residual_layers: int = 6,
    ):
        super().__init__()
        self.conv_input = ConvBlock(in_channels, residual_channels, 3)
        self.residual_tower = nn.Sequential(
            *[
                ResBlock(residual_channels, residual_channels)
                for _ in range(residual_layers)
            ]
        )
        self.policy_conv = ConvBlock(residual_channels, 2, 1)
        self.policy_fc = nn.Linear(
            1776, 4672
        )

    def forward(self, planes):
        # first conv layer
        x = self.conv_input(planes)

        # residual tower
        x = self.residual_tower(x)

        # policy head
        pol = self.policy_conv(x)
        pol = self.policy_fc(torch.flatten(pol, start_dim=1))

        return pol


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
