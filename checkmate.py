from pettingzoo.classic.chess import chess_utils


def stockfish2pettingzoo(move):
    """
    Stockfish comes in the regex format ^\d[a-z]\d[a-z]$ - string of
    length 4, 1st & 3rd chars are lowercase letters, 2nd & 4th chars
    are numbers.

    Process:
        - Stockfish looks at the current state and produces an ideal move
        in the format ^[a-z]\d[a-z]\d$ (UCI). Stockfish already has built-in
        checking for legal moves, therefore, the method doesn't need to
        check.

        - This method converts the UCI into an action value.

        - That action value is played on the Stockfish agent's behalf.

    :param move UCI format string of this ^\d[a-z]\d[a-z]$

    :return: an integer representing the index of the action to be played.
    """

    # This method makes a move-mapping and inserts it into the two
    # dictionaries.
    assert isinstance(move, str), f"Move is not a string, got {move} instead"
    chess_utils.make_move_mapping(uci_move=move)
    print(chess_utils.actions_to_moves)
    action = None
    for act, uci in chess_utils.actions_to_moves.items():
        if uci == move:
            action = act

    assert isinstance(action, int), f"action should be an integer, got {action} instead"

    return action


class Checkmate:
    """
    Checkmate - a python class used for converting Stockfish python API
    notation to Pettingzoo chess environment action space, and vice versa.
    """

    def pettingzoo2stockfish(self):
        pass
