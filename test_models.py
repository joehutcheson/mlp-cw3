import testing_torch_model
from reward_functions import *
from pettingzoo.classic import chess_v5 as chess

m = '/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish'
d = '/home/s1951999/stockfish_15.1_linux_x64/stockfish-ubuntu-20.04-x86-64'

models = [
    {
        'name': 'castling',
        'func': castling_reward
    },
    {
        'name': 'ctrl_of_centre',
        'func': control_of_centre_reward
    },
    {
        'name': 'material_advantage',
        'func': material_advantage_reward
    },
    {
        'name': 'mobility',
        'func': mobility_reward
    },
    {
        'name': 'piece_capture',
        'func': piece_capture_reward
    },
    {
        'name': 'outcome',
        'func': outcome_reward
    }
]

for m0 in models:
    for m1 in models:
        if m0 != m1:
            env = chess.env()
            env.reset()
            model = testing_torch_model.Model(env, reward_function=m0['func'],
                                              reward_function_2=m1['func'])
            model.load_model('player_0', 'models/', m0['name'])
            model.load_model('player_1', 'models/', m1['name'])
            print(f"player_0: {m0['name']} vs player_1: {m1['name']}")
            model.test()

print('Complete')
