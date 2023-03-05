__author__ = "Callum Alexander"
__email__ = "s1931801@ed.ac.uk"

import chess
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
    TODO - not tested
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

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: float("inf")
    }

    agent = env.agents.index(env.agent_selection)  # Gets the agent (0 | 1)
    board = getattr(env.unwrapped.unwrapped.unwrapped, 'board')
    assert isinstance(board, chess.Board)
    last_move = board.peek()
    assert isinstance(last_move, chess.Move)
    if board.is_capture(last_move):
        # Get the captured piece type
        captured_piece = board.piece_at(last_move.to_square)
        return piece_values[captured_piece]
    else:
        return 0


def material_advantage_reward(env: OrderEnforcingWrapper) -> int:
    """
     A positive reward signal for having a material advantage over the opponent.
    :param env: OrderEnforcingWrapper instance representing the chess
                environment
    :return: int representing the reward at the given state
    """
    white_count, black_count = 0, 0
    from chess import SQUARES
    agent = env.agents.index(env.agent_selection)  # Gets the agent (0 | 1)
    board = getattr(env.unwrapped.unwrapped.unwrapped, 'board')
    assert isinstance(board, chess.Board)
    # board.set_fen("r2qkbnr/ppp2ppp/2n5/3p1b2/2PPN2P/8/PP2PPP1/RNBQKB1R")
    for sq in SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            if piece.color == chess.WHITE:
                white_count += 1
            else:
                black_count += 1

    return white_count - black_count if agent == 0 \
        else black_count - white_count


def mobility_reward(env: OrderEnforcingWrapper) -> int:
    """
    A positive reward signal for having a greater number of available moves.
    :param env: OrderEnforcingWrapper instance representing the chess
                environment
    :return: int representing the reward at the given state
    """
    agent = env.agents.index(env.agent_selection)  # Gets the agent (0 | 1)
    board = getattr(env.unwrapped.unwrapped.unwrapped, 'board')
    assert isinstance(board, chess.Board)
    legal_moves = chess_utils.legal_moves(board)
    return len(legal_moves)


def control_of_centre_reward(env: OrderEnforcingWrapper) -> float:
    """
    # TODO - test test test this function
    A positive reward signal for controlling the central squares of the board.
    :param env: OrderEnforcingWrapper instance representing the chess
                environment
    :return: int representing the reward at the given state
    """
    agent = env.agents.index(env.agent_selection)  # Gets the agent (0 | 1)
    board = getattr(env.unwrapped.unwrapped.unwrapped, 'board')
    assert isinstance(board, chess.Board)

    # Define the central squares of the board
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]

    # Get the number of pieces each player has controlling the central squares
    white_control = sum([1 for sq in center_squares if
                         board.piece_at(sq) and board.piece_at(
                             sq).color == chess.WHITE])
    black_control = sum([1 for sq in center_squares if
                         board.piece_at(sq) and board.piece_at(
                             sq).color == chess.BLACK])

    # Calculate the central control valuation as the difference between the
    # number of pieces controlling the central squares for each player
    central_control_valuation = white_control - black_control

    return central_control_valuation


def king_safety_reward(env: OrderEnforcingWrapper) -> int:
    """
    # TODO - test test test
    A positive reward signal for keeping the king protected.
    :param env: OrderEnforcingWrapper instance representing the chess
                environment
    :return: int representing the reward at the given state
    """

    agent = env.agents.index(env.agent_selection)  # Gets the agent (0 | 1)
    board = getattr(env.unwrapped.unwrapped.unwrapped, 'board')
    assert isinstance(board, chess.Board)

    # Get the position of the kings on the board
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)

    # Check if each king is in check
    white_king_in_check = board.is_check() and board.turn == chess.WHITE
    black_king_in_check = board.is_check() and board.turn == chess.BLACK

    # Calculate the number of pieces defending each king
    white_defenders = len(board.attackers(chess.WHITE, white_king_sq))
    black_defenders = len(board.attackers(chess.BLACK, black_king_sq))

    # Calculate the king safety score as the difference between the number of
    # defenders for each king
    king_safety_score = white_defenders - black_defenders

    # Adjust the king safety score if a king is in check
    if white_king_in_check:
        king_safety_score += -1
    elif black_king_in_check:
        king_safety_score += 1

    return king_safety_score

