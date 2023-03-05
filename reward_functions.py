__author__ = "Callum Alexander"
__email__ = "s1931801@ed.ac.uk"

import chess as pychess
from pettingzoo.utils import OrderEnforcingWrapper
from pettingzoo.classic.chess import chess_utils


class RewardFunction:

    def __init__(self, env_, agent_):
        self.env = env_
        self.agent = agent_

    def get_reward(self):
        raise NotImplementedError()


def piece_capture_reward(env: OrderEnforcingWrapper) -> int:
    """
    A positive reward signal for capturing opponent's pieces.
    agent makes move
    checks for reward
        check that the previous move was legal
        check available pieces on the board,
            if material change  how do you check material change?
                previous move resulted in a taking of material
            reward
    :param env: OrderEnforcingWrapper instance representing the chess
                environment
    :return: int representing the reward at the given state
    """
    pass


def material_advantage_reward(env: OrderEnforcingWrapper) -> int:
    """
     A positive reward signal for having a material advantage over the opponent.
    :param env: OrderEnforcingWrapper instance representing the chess
                environment
    :return: int representing the reward at the given state
    """
    white_count, black_count = 0, 0

    return 0


def mobility_reward(env: OrderEnforcingWrapper) -> int:
    """
    A positive reward signal for having a greater number of available moves.
    :param env: OrderEnforcingWrapper instance representing the chess
                environment
    :return: int representing the reward at the given state
    """
    agent = env.agent_selection
    board = getattr(env.unwrapped.unwrapped.unwrapped, 'board')
    legal_moves = chess_utils.legal_moves(board)

    return 0


def control_of_centre_reward(env: OrderEnforcingWrapper) -> int:
    """
    A positive reward signal for controlling the central squares of the board.
    :param env: OrderEnforcingWrapper instance representing the chess
                environment
    :return: int representing the reward at the given state
    """
    return 0
