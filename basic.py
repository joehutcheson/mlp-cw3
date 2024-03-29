import testing_torch_model
from reward_functions import *
from pettingzoo.classic import chess_v5 as chess

env = chess.env()
env.reset()

m = '/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish'
d = '/home/s1951999/stockfish_15.1_linux_x64/stockfish-ubuntu-20.04-x86-64'

model = testing_torch_model.Model(env, reward_function=material_advantage_reward,
                                  reward_function_2=material_advantage_reward)

model.load_model('player_0', 'models/', 'material_advantage')
model.load_model('player_1', 'models/', 'material_advantage')
model.test(5)