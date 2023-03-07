import testing_torch_model
from reward_functions import *
from pettingzoo.classic import chess_v5 as chess

env = chess.env()
env.reset()


model = testing_torch_model.Model(env, reward_function=material_advantage_reward,
                                  stockfish_path='/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish',
                                  stockfish_difficulty=10)

model.train(10)