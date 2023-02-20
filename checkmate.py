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

    def stockfish2pettingzoo(self, move, agent):
        """
        Stockfish comes in the regex format ^\d[a-z]\d[a-z]$ - string of
        length 4, 1st & 3rd chars are numbers, 2nd & 4th chars are lowercase
        letters  TODO - Fix this shit it's letter number, not number letter

        :param move - string of this ^\d[a-z]\d[a-z]$

        :return: a standard basis vector of size 4672 representing the output
        action to be fed to the environment (8, 8, 73)
        """

        if len(move) != 4:
            raise Exception(f"Unrecognised move format? {move}")

        start = np.array([ord(move[0]) - 97, int(move[1])])  # []
        finish = np.array([ord(move[2]) - 97, int(move[3])])

        output = 73 * 8 * (start[1] if agent == 'player_0' else 8 - start[1]) + 73 * start[0]





        change_pos = finish - start

    def pettingzoo2stockfish(self):
        pass
