import time
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta, Adam, SGD
from keras.utils import np_utils
from keras import backend as K
from keras_tqdm import TQDMNotebookCallback
from keras.callbacks import TensorBoard
import sys
import random

INITIAL_SHAPE = (3, 15, 15)
INITIAL_ROLLOUT_SHAPE = (27, 15, 15)


def get_rollout_result(board, last_action):  # kind of optimized
    if last_action is None:
        return None
    row, col = last_action[0], last_action[1]
    val = board[row, col]
    if val == 0.0:
        return None
    counts = 1
    for new_row in range(row+1, 15):
        if board[new_row, col] == val:
            counts += 1
        else:
            break
    for new_row in range(row-1, -1, -1):
        if board[new_row, col] == val:
            counts += 1
        else:
            break
    if counts >= 5:
        return val

    counts = 1
    for new_col in range(col + 1, 15):
        if board[row, new_col] == val:
            counts += 1
        else:
            break
    for new_col in range(col - 1, -1, -1):
        if board[row, new_col] == val:
            counts += 1
        else:
            break
    if counts >= 5:
        return val

    counts = 1
    for new_row, new_col in zip(range(row+1, 15), range(col+1, 15)):
        if board[new_row, new_col] == val:
            counts += 1
        else:
            break
    for new_row, new_col in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
        if board[new_row, new_col] == val:
            counts += 1
        else:
            break
    if counts >= 5:
        return val

    counts = 1
    for new_row, new_col in zip(range(row + 1, 15), range(col - 1, -1, -1)):
        if board[new_row, new_col] == val:
            counts += 1
        else:
            break
    for new_row, new_col in zip(range(row - 1, -1, -1), range(col + 1, 15)):
        if board[new_row, new_col] == val:
            counts += 1
        else:
            break
    if counts >= 5:
        return val

    if np.abs(board).sum() == board.size:
        return 0.0
    else:
        return None

def transform_th(board):
    return board

def transform_tf(board):
    return board.transpose((1, 2, 0))

def move_to_indices(string):
    width = ord(string[0])
    if (width > ord('i')):
        width -= 1
    width -= ord('a')
    height = int(string[1:]) - 1
    return (height, width)

def indices_to_move(indices):
    width = ord('a') + indices[1]
    if width >= ord('i'):
        width += 1
    width = chr(width)
    height = str(indices[0] + 1)
    return width + height

if K.image_dim_ordering() == 'th':  # some hard-code ^_^
    INPUT_SHAPE = INITIAL_SHAPE
    INPUT_ROLLOUT_SHAPE = INITIAL_ROLLOUT_SHAPE
    transform_board = transform_th
else:
    INPUT_SHAPE = (INITIAL_SHAPE[1], INITIAL_SHAPE[2], INITIAL_SHAPE[0])
    INPUT_ROLLOUT_SHAPE = (INITIAL_ROLLOUT_SHAPE[1], INITIAL_ROLLOUT_SHAPE[2], INITIAL_ROLLOUT_SHAPE[0])
    transform_board = transform_tf

def generate_games(filename):
    with open(filename, 'r') as file:
        for line in file:
            yield Game(line)
            
def generate_value_games(filename):
    with open(filename, 'r') as file:
        for line in file:
            game = Game(line)
            if game.result == -1 or game.result == 0 or game.result == 1:
                if len(game.turns) >= 1:
                    yield Game(line)

class Game:
    def __init__(self, string):
        if string.startswith("draw"):
            self.result = 0
            string = string[5:]
        elif string.startswith("black"):
            self.result = -1
            string = string[6:]
        elif string.startswith("white"):
            self.result = 1
            string = string[6:]
        elif string.startswith("unknown"):
            self.result = 2
            string = string[8:]
        string = string[:-1]
        self.turns = string.split()

    def generate_actions(self, last_n=None):
        board = np.zeros((15, 15), dtype='float')
        for i in range(len(self.turns)):
            width = ord(self.turns[i][0])
            if width > ord('i'):
                width -= 1
            width -= ord('a')
            height = int(self.turns[i][1:]) - 1
            is_black = (i % 2 == 0)
            for j in range(2):
                for rot in range(4):
                    if last_n is not None and len(self.turns) - i > last_n:
                        continue
                    res = (prepare_board(board, player_black=is_black),
                           np_utils.to_categorical(np.array([height*15 + width]), num_classes=225))
                    yield res
                    board = np.rot90(board)
                    height, width = 14 - width, height
                board = np.fliplr(board)
                width = 14 - width
            board[height, width] = -1 + 2*(i % 2)

    def generate_value_actions(self, last_n=None):
        board = np.zeros((15, 15), dtype='float')
        for i in range(len(self.turns)):
            width = ord(self.turns[i][0])
            if (width > ord('i')):
                width -= 1
            width -= ord('a')
            height = int(self.turns[i][1:]) - 1
            for j in range(2):
                for rot in range(4):
                    if last_n is not None and len(self.turns) - i > last_n:
                        continue
                    yield (prepare_board(board), float(self.result))
                    board = np.rot90(board)
                    height, width = 14 - width, height
                board = np.fliplr(board)
                width = 14 - width
            board[height, width] = -1 + 2*(i % 2)

    def generate_value_rand_action(self):
        board = np.zeros((15, 15), dtype='float')
        turn = np.random.randint(0, len(self.turns))
        for i in range(len(self.turns)):
            width = ord(self.turns[i][0])
            if width > ord('i'):
                width -= 1
            width -= ord('a')
            height = int(self.turns[i][1:]) - 1
            if i == turn:
                for j in range(2):
                    for rot in range(4):
                        for state in [(False, 1.0), (True, -1.0)]:
                            yield (prepare_board(board, state[0]), state[1] * self.result)
                        board = np.rot90(board)
                        height, width = 14 - width, height
                    board = np.fliplr(board)
                    width = 14 - width
                raise StopIteration
            board[height, width] = -1 + 2*(i % 2)


def legal_nn_move(board, policy, random=False):
    is_black = (int(board.sum()) == 0)
    p_board = prepare_board(board,
                            is_black)
    probas = policy.predict_proba(p_board[np.newaxis], verbose=0)[0]
    probas += 0.00001
    mask = 1 - np.abs(board).flatten()
    probas *= mask
    if not random:
        action = np.argmax(probas)
    else:
        probas /= probas.sum()
        action = np.random.choice(225, p=probas)
    return action // 15, action % 15


def generate_reinforced_samples(sl_policy, rl_policy, max_turns):
    while True:
        board = np.zeros((15, 15), dtype='float')
        turns = int(np.random.uniform(0, max_turns))
        last_action = (0, 0)
        res = None
        restart = False
        for turn in range(turns):
            res = get_rollout_result(board, last_action)
            if res is not None:
                restart = True
                break
            last_action = legal_nn_move(board, sl_policy)
            board[last_action] = -1.0 + 2 * (turn % 2)
        turn = turns
        if restart:
            continue
        res = get_rollout_result(board, last_action)
        if res is not None:
            continue
        mask = 1.0 - np.abs(board).flatten()
        mask /= mask.sum()
        last_action = np.random.choice(225, p=mask)
        last_action = (last_action // 15, last_action % 15)
        board[last_action] = -1.0 + 2 * (turn % 2)
        turn += 1
        board_in_interest = board.copy()
        while get_rollout_result(board, last_action) is None:
            last_action = legal_nn_move(board, rl_policy)
            board[last_action] = -1.0 + 2 * (turn % 2)
            turn += 1
        res = get_rollout_result(board, last_action)
        return ((prepare_board(board_in_interest, player_black=False), res),
                (prepare_board(board_in_interest, player_black=True), -res))


    
def show_board(board, show=True, size=7):
    ticks = np.array((range(0, 15)))
    ylabels = ticks+1
    xlabels = [chr(x) for x in list(range(ord('a'), ord('i'))) + list(range(ord('j'), ord('z')+1))][:15]
    if show:
        plt.figure(figsize=(size, size))
    plt.imshow(board, cmap='hot', interpolation='nearest')
    plt.yticks(ticks, ylabels, size=round(size*2.5))
    plt.xticks(ticks, xlabels, size=round(size*2.5))
    plt.grid(True)
    if show:
        plt.show()


def prepare_board(board, player_black=None):
    prepared_board = np.zeros(INITIAL_SHAPE, dtype='float32')
    player_value = -1.0 if player_black else 1.0
    prepared_board[0][board == player_value] = 1.0
    prepared_board[1][board == -player_value] = 1.0
    prepared_board[2][board == 0.0] = 1.0
    return transform_board(prepared_board)


def get_selfplay_length(model):
    last_action = (0, 0)
    board = np.zeros((15, 15))
    turn = 0
    while get_rollout_result(board, last_action) is None:
        last_action = legal_nn_move(board, model)
        board[last_action] = -1.0 + 2 * (turn % 2)
        turn += 1
    return turn


def generate_batch(filename, batch_size, value_nn=False, last_n=None):
    if value_nn:
        games_generator = generate_value_games(filename)
    else:
        games_generator = generate_games(filename)
    actions = 0
    if not value_nn:
        batch_samples = np.zeros((batch_size, *INPUT_SHAPE), dtype='float32')  # TODO: refactor hard-code
        batch_labels = np.zeros((batch_size, 225), dtype='float32')
    else:
        batch_samples = np.zeros((batch_size, *INPUT_SHAPE), dtype='float32')  # TODO: refactor hard-code
        batch_labels = np.zeros((batch_size, 1), dtype='float32') # TODO: remember: may strike back in the future
    index = 0
    while True:
        for game in games_generator:
            if value_nn:
                actions_generator = game.generate_value_actions(last_n=last_n)
            else:
                actions_generator = game.generate_actions(last_n=last_n)
            for action_idx, action in enumerate(actions_generator):
                batch_samples[index], batch_labels[index] = action[0], action[1]
                index += 1
                index %= batch_size
                actions += 1
                if index == 0:
                    yield (batch_samples, batch_labels)
                    if not value_nn:
                        batch_samples = np.zeros((batch_size, *INPUT_SHAPE), dtype='float32')
                        batch_labels = np.zeros((batch_size, 225), dtype='float32')
                    else:
                        batch_samples = np.zeros((batch_size, *INPUT_SHAPE), dtype='float32')
                        batch_labels = np.zeros((batch_size, 1), dtype='float32')


def generate_reinforced_batch(sl_policy, rl_policy, batch_size):
    batch_samples = np.zeros((batch_size, *INPUT_SHAPE), dtype='float32')  # TODO: refactor hard-code
    batch_labels = np.zeros((batch_size, 225), dtype='float32')
    index = 0
    while True:
        for state in generate_reinforced_samples(sl_policy, rl_policy):
            batch_samples[index], batch_labels[index] = state[0], state[1]
            index += 1
            index %= batch_size
            if index == 0:
                yield (batch_samples, batch_labels)
                
                
def measure_dataset(filename, mark_player=False, last_n=None, nb_games=None):
    if mark_player:
        games_generator = generate_value_games(filename)
    else:
        games_generator = generate_games(filename)
    index = 0
    total = 0
    for game in tqdm(games_generator):
        if nb_games is None or total < nb_games:
            if mark_player:
                actions_generator = game.generate_value_actions(last_n=last_n)
            else:
                actions_generator = game.generate_actions(last_n=last_n)
            for action in actions_generator:
                index += 1
        total += 1
    if nb_games is not None and total > nb_games:
        index = int(index * total/nb_games)
    return index