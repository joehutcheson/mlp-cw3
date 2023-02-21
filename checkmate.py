import numpy as np
from pettingzoo.classic.chess import chess_utils


import stockfish


class Checkmate:
    """
    Checkmate - a python class used for converting Stockfish python API
    notation to Pettingzoo chess environment action space, and vice versa.
    """

    def __init__(self):
        pass

    def stockfish2pettingzoo(self, move):
        """
        Stockfish comes in the regex format ^\d[a-z]\d[a-z]$ - string of
        length 4, 1st & 3rd chars are numbers, 2nd & 4th chars are lowercase
        letters  TODO - Fix this shit it's letter number, not number letter

        :param move - UCI format string of this ^\d[a-z]\d[a-z]$

        :return: a standard basis vector of size 4672 representing the output
        action to be fed to the environment (8, 8, 73)
        """

        action = chess_utils.actions_to_moves[move]
        return action



    def pettingzoo2stockfish(self):
        pass
