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
        env.step(checkmate.stockfish2pettingzoo(
            chess.Move(chess.E2, chess.E4).__str__()))

        print(checkmate.stockfish2pettingzoo(
            chess.Move(chess.D7, chess.D5).__str__()))

        print(checkmate.pettingzoo2stockfish(env, 2201))
        # env.step(checkmate.stockfish2pettingzoo(
        #     chess.Move(chess.D7, chess.D5).__str__()))

        print(env.render())
        # env.step(checkmate.stockfish2pettingzoo(
        #     chess.Move(chess.E4, chess.D5).__str__()))
        #
        # # Check that the reward is 1 (the value of a captured pawn)
        # self.assertEqual(piece_capture_reward(env), 1)


if __name__ == '__main__':
    unittest.main()
