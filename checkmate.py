import chess
from pettingzoo.classic.chess import chess_utils
from pettingzoo.classic.chess.chess import raw_env
from pettingzoo.utils.wrappers import OrderEnforcingWrapper


def stockfish2pettingzoo(env: OrderEnforcingWrapper, move: str) -> int:
    """
    TODO implement mirroring for promotions
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

    :param env:
    :param move UCI format string of this ^\d[a-z]\d[a-z]$

    :return: an integer representing the index of the action to be played.
    """

    # This method makes a move-mapping and inserts it into the two
    # dictionaries.
    assert isinstance(move,
                      str), f"[-] Move is not a string, got {move} instead"

    env_u = env.unwrapped.unwrapped.unwrapped
    assert isinstance(env_u,
                      raw_env), f"[-] given environment can't be reduced " \
                                f"to a raw_env instance, got " \
                                f"{type(env_u)} instead"

    player = bool(env.agents.index(env.agent_selection))

    base_move = chess.Move.from_uci(uci=move)
    mirr_move = chess_utils.mirror_move(base_move) if player else base_move
    #
    chess_utils.make_move_mapping(uci_move=mirr_move.uci())
    action = chess_utils.moves_to_actions[mirr_move.uci()]
    assert isinstance(action,
                      int), f"[-] action should be an integer, got {action}"
    return action


def pettingzoo2stockfish(env: OrderEnforcingWrapper, action_value: int) -> str:
    """
    This function takes in the current environmnet of the game and an integer
    representing the action_value produced by the DQN agent. This function
    then returns the corresponding uci value for the action value.

    This uci value is passed into the Stockfish API when Stockfish recognises
    that it's the DQN's move.
    :param env: OrderEnforcingWrapper instance representing the chess
                environment
    :param action_value: An integer representing the action value produced from
                         the DQN, representing its own move
    :return: a string representing the action value as a uci move
    """

    env_u = env.unwrapped.unwrapped.unwrapped
    assert isinstance(env_u,
                      raw_env), f"[-] given environment can't be reduced " \
                                f"to a raw_env instance, got " \
                                f"{type(env_u)} instead"

    orig_board = getattr(env_u, "board")
    lm = chess_utils.legal_moves(orig_board=orig_board)  # This should make a
    # move mapping
    assert action_value in lm, f"[-] Action `{action_value}` not in legal " \
                               f"moves, `{lm}`"  # check action_value in
    # legal moves, otherwise this function is screwed
    player = bool(env.agents.index(env.agent_selection))
    act_uci = None
    try:
        act_uci = chess_utils.action_to_move(orig_board, action_value, player)
    except:
        print(f"[-] {action_value} may not be legal")
    return act_uci.uci()
