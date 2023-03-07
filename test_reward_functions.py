import unittest

import checkmate
from reward_functions import *
from pettingzoo.utils.wrappers import OrderEnforcingWrapper
from pettingzoo.classic import chess_v5 as ch


class MyTestCase(unittest.TestCase):

    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_no_capture(self):
        # Set up the environment
        env = ch.env()
        env.reset()

        # Perform a move that doesn't capture anything
        mv = checkmate.stockfish2pettingzoo(
            chess.Move(chess.E2, chess.E3).__str__())
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
        print(env.render())
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

        print(env.render())
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

        print(env.render())
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

        print(env.render())
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

        print(env.render())
        # Perform a move that captures a queen
        env.step(checkmate.stockfish2pettingzoo(env,
                                                chess.Move(chess.B6,
                                                           chess.C7).__str__()))
        # Check that the reward is 9 (the value of a captured queen)
        self.assertEqual(piece_capture_reward(env), 5)


if __name__ == '__main__':
    unittest.main()
