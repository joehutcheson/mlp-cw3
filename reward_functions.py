__author__ = "Callum Alexander"
__email__ = "s1931801@ed.ac.uk"

import chess
import chess as pychess
from pettingzoo.utils import OrderEnforcingWrapper
from pettingzoo.classic.chess import chess_utils
import numpy as np


# ---- Normalisation Functions ----
def normalize_rewards(reward_list):
    """
    Normalizes a list of reward signals using min-max scaling and returns a
    single aggregated reward signal.
    :param: reward_list: List of reward signals
    :return aggregated_reward: Single normalised reward
    """
    normalized_rewards = []
    for rewards in reward_list:
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        if max_reward == min_reward:
            normalized_rewards.append(np.array(rewards))
        else:
            normalized_rewards.append(
                (np.array(rewards) - min_reward) / (max_reward - min_reward))
    aggregated_reward = np.mean(normalized_rewards, axis=0)
    return aggregated_reward


# ---- Reward Functions ----
class RewardFunction:

    def __init__(self, env_, agent_):
        self.env = env_
        self.agent = agent_

    def get_reward(self):
        raise NotImplementedError()


def piece_capture_reward(env: OrderEnforcingWrapper) -> int:
    """
    **Test covered**\n
    A positive reward signal for capturing opponent's pieces.
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

    last_board = board.copy()  # Makes a copy
    last_move = last_board.pop()  # Undoes the last move and assigns it
    assert isinstance(last_move, chess.Move)
    assert last_board.fen() != board.fen()

    if last_board.is_capture(last_move):
        # Get the captured piece type
        captured_piece = last_board.piece_at(last_move.to_square).piece_type
        last_board = board.copy()
        return piece_values[captured_piece]
    else:
        last_board = board.copy()
        return 0


def castling_reward(env: OrderEnforcingWrapper) -> int:
    """
    **Test covered**\n
    Todo documentation
    :param env:
    :return:
    """
    agent = env.agents.index(env.agent_selection)  # Gets the agent (0 | 1)
    board = getattr(env.unwrapped.unwrapped.unwrapped, 'board')
    assert isinstance(board, chess.Board)

    if board.move_stack:
        # Get the last move in the move stack
        last_board = board.copy()  # Makes a copy
        last_move = last_board.pop()  # Undoes the last move and assigns it
        assert isinstance(last_move, chess.Move)
        assert last_board.fen() != board.fen()

        # Check if the last move was a castling move
        if last_board.is_castling(last_move):
            # Get the color of the player who made the move
            player_color = last_board.turn

            # Determine which side the player castled on
            if last_move.to_square == chess.G1:
                # Kingside castle for white
                if player_color == chess.WHITE:
                    return 1
            elif last_move.to_square == chess.C1:
                # Queenside castle for white
                if player_color == chess.WHITE:
                    return 1
            elif last_move.to_square == chess.G8:
                # Kingside castle for black
                if player_color == chess.BLACK:
                    return 1
            elif last_move.to_square == chess.C8:
                # Queenside castle for black
                if player_color == chess.BLACK:
                    return 1

    # Return zero if the last move was not a castling move
    return 0


def material_advantage_reward(env: OrderEnforcingWrapper) -> int:
    """
    A positive (or negative) reward signal for having a material advantage over
    the opponent.
    :param env: OrderEnforcingWrapper instance representing the chess
                environment
    :return: int representing the reward at the given state
    """
    white_count, black_count = 0, 0
    from chess import SQUARES
    agent = env.agents.index(env.agent_selection)  # Gets the agent (0 | 1)
    board = getattr(env.unwrapped.unwrapped.unwrapped, 'board')
    assert isinstance(board, chess.Board)
    for sq in SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            if piece.color == chess.WHITE:
                white_count += 1
                assert white_count <= 16, f"[-] got more white pieces than " \
                                          f"expected, c:{white_count} sq:{sq} t:{piece.piece_type}"
            elif piece.color == chess.BLACK:
                black_count += 1
                assert black_count <= 16, f"[-] got more black pieces than " \
                                          f"expected, {black_count}  sq:{sq} t:{piece.piece_type}"

    x = 1

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
    central_control_valuation = white_control - black_control if agent == 0 \
        else black_control - white_control

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


def pawn_structure_reward(env: OrderEnforcingWrapper) -> int:
    """
    A positive reward signal for having a strong pawn structure.
    Heavy lifting done by evaluate_pawns() helper function.
    TODO - not tested
    :param env: OrderEnforcingWrapper instance representing the chess
                environment
    :return: int representing the reward at the given state
    """

    agent = env.agents.index(env.agent_selection)  # Gets the agent (0 | 1)
    board = getattr(env.unwrapped.unwrapped.unwrapped, 'board')
    assert isinstance(board, chess.Board)
    # Initialize variables to store pawn counts and scores for each player
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)
    white_pawn_count = len(white_pawns)
    black_pawn_count = len(black_pawns)
    white_pawn_score = 0
    black_pawn_score = 0

    # Define a helper function to calculate pawn structure scores for each
    # player
    def evaluate_pawns(pawns, color):
        """
        Helper function for evaluating the structure of a player's pawns.
        Scores 3 heuristics for determining good or bad pawn structure:

        - Isolated pawn: An isolated pawn in chess is a pawn that has no other pawns of the same color on adjacent files.
        - Doubled pawn: A doubled pawn in chess is a situation where two pawns of the same color are on the same file, one behind the other.
        - Passed pawn: A passed pawn in chess is a pawn that has no opposing pawns on its path to promotion which means it has the potential to advance and become a queen.

        :param pawns: SquareSet representing the list of squares filled by
        pawns of a certain colour
        :param color: Boolean value representing
        the colour of the pawns being played. White == True, False == Black
        :return: pawn structure score
        """
        score = 0
        for pawn_sq in pawns:
            pawn_file = chess.square_file(pawn_sq)
            pawn_rank = chess.square_rank(pawn_sq)

            # Check if the pawn is isolated
            is_isolated = True
            for file_offset in (-1, 1):
                if (0 <= pawn_file + file_offset <= 7 and
                    board.piece_type_at(chess.square(pawn_file + file_offset,
                                                     pawn_rank)) == chess.PAWN and
                    board.color_at(chess.square(pawn_file + file_offset,
                                                pawn_rank)) == color):
                    is_isolated = False
                    break
            if is_isolated:
                score -= 10

            # Check if the pawn is doubled
            if board.pawns_on_file(pawn_file) > 1:
                score -= 5

            # Check if the pawn is passed
            is_passed = True
            for rank_offset in (-1, 1):
                for file_offset in (-1, 1):
                    if (0 <= pawn_file + file_offset <= 7 and
                        0 <= pawn_rank + rank_offset <= 7 and
                        board.piece_type_at(
                            chess.square(pawn_file + file_offset,
                                         pawn_rank + rank_offset)) == chess.PAWN and
                        board.color_at(chess.square(pawn_file + file_offset,
                                                    pawn_rank + rank_offset)) == color):
                        is_passed = False
                        break
                if not is_passed:
                    break
            if is_passed:
                score += 10

        return score

    # Calculate pawn structure scores for each player
    white_pawn_score = evaluate_pawns(white_pawns, chess.WHITE)
    black_pawn_score = evaluate_pawns(black_pawns, chess.BLACK)

    # Return the difference between the pawn structure scores for each player
    return white_pawn_score - black_pawn_score
