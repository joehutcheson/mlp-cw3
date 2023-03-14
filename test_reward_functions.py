import unittest

from random import randint
import random

import chess

import checkmate
from reward_functions import *
from pettingzoo.utils.wrappers import OrderEnforcingWrapper
from pettingzoo.classic import chess_v5 as ch


class TestCapturePieceReward(unittest.TestCase):

    def always_true_test(self):
        self.assertEqual(True, True)  # add assertion here

    def test_no_capture(self):
        # Set up the environment
        env = ch.env()
        env.reset()

        # Perform a move that doesn't capture anything
        mv = checkmate.stockfish2pettingzoo(env,
                                            chess.Move(chess.E2,
                                                       chess.E3).__str__())
        env.step(mv)

        # Check that the reward is 0
        self.assertEqual(piece_capture_reward(env), 0)

    def test_capture_pawn(self):
        # Set up the environment
        env = ch.env(render_mode='ansi')
        env.reset()

        # Perform a move that captures a pawn
        env.step(checkmate.stockfish2pettingzoo(env,
                                                chess.Move(chess.E2,
                                                           chess.E4).__str__()))
        env.step(checkmate.stockfish2pettingzoo(env,
                                                chess.Move(chess.D7,
                                                           chess.D5).__str__()))
        env.step(checkmate.stockfish2pettingzoo(env,
                                                chess.Move(chess.E4,
                                                           chess.D5).__str__()))
        # Check that the reward is 1 (the value of a captured pawn)
        self.assertEqual(piece_capture_reward(env), 1)

    def test_capture_queen(self):
        # Set up the environment
        env = ch.env(render_mode='ansi')
        env.reset()
        board = getattr(env.unwrapped.unwrapped.unwrapped, 'board')
        board.set_fen("1k6/2q5/1P6/8/8/8/8/4K3 w - - 0 1")

        # Perform a move that captures a queen
        env.step(checkmate.stockfish2pettingzoo(env,
                                                chess.Move(chess.B6,
                                                           chess.C7).__str__()))
        # Check that the reward is 9 (the value of a captured queen)
        self.assertEqual(piece_capture_reward(env), 9)

    def test_capture_bishop(self):
        # Set up the environment
        env = ch.env(render_mode='ansi')
        env.reset()
        board = getattr(env.unwrapped.unwrapped.unwrapped, 'board')
        board.set_fen("1k6/2b5/1P6/8/8/8/8/4K3 w - - 0 1")

        # Perform a move that captures a queen
        env.step(checkmate.stockfish2pettingzoo(env,
                                                chess.Move(chess.B6,
                                                           chess.C7).__str__()))
        # Check that the reward is 9 (the value of a captured queen)
        self.assertEqual(piece_capture_reward(env), 3)

    def test_capture_knight(self):
        # Set up the environment
        env = ch.env(render_mode='ansi')
        env.reset()
        board = getattr(env.unwrapped.unwrapped.unwrapped, 'board')
        board.set_fen("1k6/2n5/1P6/8/8/8/8/4K3 w - - 0 1")

        # Perform a move that captures a queen
        env.step(checkmate.stockfish2pettingzoo(env,
                                                chess.Move(chess.B6,
                                                           chess.C7).__str__()))
        # Check that the reward is 9 (the value of a captured queen)
        self.assertEqual(piece_capture_reward(env), 3)

    def test_capture_rook(self):
        # Set up the environment
        env = ch.env(render_mode='ansi')
        env.reset()
        board = getattr(env.unwrapped.unwrapped.unwrapped, 'board')
        board.set_fen("1k6/2r5/1P6/8/8/8/8/4K3 w - - 0 1")

        # Perform a move that captures a queen
        env.step(checkmate.stockfish2pettingzoo(env,
                                                chess.Move(chess.B6,
                                                           chess.C7).__str__()))
        # Check that the reward is 9 (the value of a captured queen)
        self.assertEqual(piece_capture_reward(env), 5)


class TestCastlingReward(unittest.TestCase):

    def test_always_true(self):
        self.assertEqual(True, True)

    def test_castling(self):
        env = ch.env(render_mode='ansi')
        env.reset()
        board = getattr(env.unwrapped.unwrapped.unwrapped, 'board')
        board.set_fen("rnbqkbnr/pppppppp/8/8/8/4PN2/PPPP1PPP/RNBQK2R w KQkq - "
                      "0 1")

        env.step(checkmate.stockfish2pettingzoo(env, chess.Move.from_uci(
            'e1g1').__str__()))

        self.assertEqual(castling_reward(env), 1)

    def test_not_castling_move(self):
        env = ch.env(render_mode='ansi')
        env.reset()
        board = getattr(env.unwrapped.unwrapped.unwrapped, 'board')
        board.set_fen("1k6/2r5/1P6/8/8/8/8/4K3 w - - 0 1")

        # Perform a move that captures a queen
        env.step(checkmate.stockfish2pettingzoo(env,
                                                chess.Move(chess.B6,
                                                           chess.C7).__str__()))

        self.assertEqual(castling_reward(env), 0)


class TestMaterialAdvantageReward(unittest.TestCase):

    def setUp(self) -> None:
        self.env = ch.env(render_mode='ansi')

    def test_no_advantage(self) -> None:
        self.env.reset()
        self.assertEqual(material_advantage_reward(self.env), 0)

    def test_white_advantage(self) -> None:
        self.env.reset()
        board = getattr(self.env.unwrapped.unwrapped.unwrapped, 'board')
        board.set_fen('k7/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1')
        self.assertEqual(15, material_advantage_reward(self.env))

    def test_black_advantage(self) -> None:
        self.env.reset()
        board = getattr(self.env.unwrapped.unwrapped.unwrapped, 'board')
        board.set_fen('K7/8/8/8/8/8/pppppppp/rnbqkbnr b KQ - 0 1')
        self.assertEqual(-15, material_advantage_reward(self.env))


def generate_random_fen_with_legal_moves_helper():
    board = chess.Board()
    for x in range(0, randint(0, 100)):
        # Make a random legal move
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            # If no legal moves left, start over
            board.reset()
            continue
        random_move = random.choice(legal_moves)
        board.push(random_move)

    return board.fen(), len(list(board.legal_moves))


class TestMobilityReward(unittest.TestCase):

    def setUp(self) -> None:
        self.env = ch.env(render_mode='ansi')

    def test_white_starting(self) -> None:
        self.env.reset()
        board = getattr(self.env.unwrapped.unwrapped.unwrapped, 'board')
        board.set_fen(
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        self.assertEqual(20, mobility_reward(self.env))

    def test_black_starting(self) -> None:
        self.env.reset()
        board = getattr(self.env.unwrapped.unwrapped.unwrapped, 'board')
        board.set_fen(
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1')
        self.assertEqual(20, mobility_reward(self.env))

    def test_random_state(self) -> None:
        for x in range(0, 1000):
            self.env.reset()
            board = getattr(self.env.unwrapped.unwrapped.unwrapped, 'board')
            fen_, lms = generate_random_fen_with_legal_moves_helper()
            board.set_fen(fen_)
            self.assertEqual(lms, mobility_reward(self.env))


def generate_random_fen_with_centre_control():
    board = chess.Board()
    for x in range(0, randint(0, 100)):
        # Make a random legal move
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            # If no legal moves left, start over
            board.reset()
            continue
        random_move = random.choice(legal_moves)
        board.push(random_move)

    # Define the central squares of the board
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]

    # Get the number of pieces each player has controlling the central squares
    white_control = sum([1 for sq in center_squares if
                         board.piece_at(sq) and board.piece_at(
                             sq).color == chess.WHITE])
    black_control = sum([1 for sq in center_squares if
                         board.piece_at(sq) and board.piece_at(
                             sq).color == chess.BLACK])

    score_to_return = white_control - black_control if \
        board.turn == chess.WHITE else black_control - white_control

    return board.fen(), score_to_return


class TestControlOfTheCentreReward(unittest.TestCase):

    def setUp(self) -> None:
        self.env = ch.env(render_mode='ansi')

    def test_random_states(self) -> None:
        # TODO fix this bcs it doesn't work and it's something to do with who's turn it is
        for _ in range(0, 5):
            self.env.reset()
            board = getattr(self.env.unwrapped.unwrapped.unwrapped, 'board')
            fen_, score = generate_random_fen_with_centre_control()
            board.set_fen(fen_)
            self.assertEqual(score, control_of_centre_reward(self.env))


if __name__ == '__main__':
    unittest.main(verbosity=1)
