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
    assert isinstance(action, int), f"action should be an integer, got {action}"
    return action


def pettingzoo2stockfish(env, action_value):
    """
    This function takes in the current environmnet of the game and an integer
    representing the action_value produced by the DQN agent. This function
    then returns the corresponding uci value for the action value.

    This uci value is passed into the Stockfish API when Stockfish recognises
    that it's the DQN's move.
    :param env: OrderEnforcingWrapper instance representing the chess
                environment
    :param action_value: An integer representing the action value produced from
                         the DQN, representing it's own move
    :return: a string representing the action value as a uci move
    """

    from pettingzoo.classic.chess.chess import raw_env

    env_u = env.unwrapped.unwrapped.unwrapped
    assert isinstance(env_u, raw_env), f"given environment can't be reduced " \
                                       f"to a raw_env instance, got " \
                                       f"{type(env_u)} instead"

    orig_board = getattr(env_u, "board")

    act = 2421  # King's pawn to e4, classic opening
    legal_moves = chess_utils.legal_moves(orig_board=orig_board)
    act_uci = chess_utils.actions_to_moves[act]

    return act_uci



