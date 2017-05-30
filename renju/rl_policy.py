import os
import random
from generate import *
from tree_search import *
from tqdm import *
import keras
import sys
from sys import stderr
import time
from keras import backend as K

def update(model, samples, labels, factor, won=None):
    old_lr = K.abs(model.optimizer.lr)
    if won is None:
        model.optimizer.lr = old_lr * (-1.5)
    else:
        if won == True:
            model.optimizer.lr = old_lr * 2.0
        else:
            model.optimizer.lr = old_lr * (-2.0)
    # initial_lr = keras.backend.get_value(model.optimizer.lr)
    # keras.backend.set_value(model.optimizer.lr, initial_lr * factor)
    model.train_on_batch(samples, labels)
    # keras.backend.set_value(model.optimizer.lr, initial_lr)
    # model.optimizer.lr = old_lr
    model.optimizer.lr = old_lr

def train(current_model, models_dir, games, games_in_batch=1):
    opponents = [load_model(os.path.join(models_dir, filename)) for filename in os.listdir(models_dir) if filename.endswith('.keras')]
    wins = 0
    losses = 0
    draws = 0
    finished_games = games
    pos_samples = []
    pos_labels = []
    neg_samples = []
    neg_labels = []
    nt_samples = []
    nt_labels = []
    for game in trange(games):
        opp_num = random.choice(range(len(opponents)))
        opponent = opponents[opp_num]
        states = []
        actions = []
        last_action = (0, 0)
        board = np.zeros((15, 15), dtype='float32')
        if game % 2 == 0:
            nets = [current_model, opponent]
        else:
            nets = [opponent, current_model]
        turn = 0
        error_occured = False  # HOLY FUCK; that should be fixed
        while get_rollout_result(board, last_action) is None:
            last_action = legal_nn_move(board, nets[turn % 2], random=False)
            action_int = last_action[0]*15 + last_action[1]
            actions.append(np_utils.to_categorical(np.array([action_int]), nb_classes=225))
            states.append(prepare_board(board, bool(1 - turn % 2)))
            board[last_action] = -1.0 + 2*(turn % 2)
            if turn >= 225:
                error_occured = True
                finished_games -= 1
                print('Some shit happened', file=stderr)
                break
            turn += 1
        if error_occured:
            continue
        samples = np.stack(states)
        labels = np.concatenate(actions)
        res = get_rollout_result(board, last_action)
        factor = res * (-1.0 + 2*(game % 2))
        if round(factor) == 1:
            pos_samples.append(samples)
            pos_labels.append(labels)
            wins += 1
            if wins % games_in_batch == 0:
                update(current_model, np.concatenate(pos_samples), np.concatenate(pos_labels), 1.0, won=True)
                pos_samples = []
                pos_labels = []
        elif round(factor) == -1:
            neg_samples.append(samples)
            neg_labels.append(labels)
            losses += 1
            if losses % games_in_batch == 0:
                update(current_model, np.concatenate(neg_samples), np.concatenate(neg_labels), -1.0, won=False)
                neg_samples = []
                neg_labels = []
        else:
            nt_samples.append(samples)
            nt_labels.append(labels)
            draws += 1
            if draws % games_in_batch == 0:
                update(current_model, np.concatenate(nt_samples), np.concatenate(nt_labels), -0.5, won=None)
                nt_samples = []
                nt_labels = []
    model_idx = len(opponents)
    current_model.save(os.path.join(models_dir, 'rl_policy_{0:0>3}.keras'.format(model_idx)))
    return wins / finished_games, losses / finished_games, draws / finished_games


def test(current_model, opponent, games):
    wins = 0
    losses = 0
    draws = 0
    for game in trange(games):
        last_action = (0, 0)
        board = np.zeros((15, 15), dtype='float32')
        if game % 2 == 0:
            nets = [current_model, opponent]
        else:
            nets = [opponent, current_model]
        turn = 0
        while get_rollout_result(board, last_action) is None:
            last_action = legal_nn_move(board, nets[turn % 2], random=True)
            board[last_action] = -1.0 + 2*(turn % 2)
            turn += 1
        res = round(get_rollout_result(board, last_action))
        factor = res * (-1 + 2 * (game % 2))
        if factor == 1:
            wins += 1
        elif factor == -1:
            losses += 1
        else:
            draws += 1
    return wins / games, losses / games, draws / games


def load_latest_model(models_dir):
    opponents = [filename for filename in os.listdir(models_dir) if filename.endswith('.keras')]
    return load_model(os.path.join(models_dir, opponents[-1]))


print('starting...')
if len(sys.argv) < 2:
    print('Usage: <script> train nb_epochs epoch_size=1250 games_in_batch=25 | test model=001 opponent=000 nb_games=1000')
    exit(0)
if sys.argv[1] == 'train':
    if len(sys.argv) != 5:
        print("Not enough arguments!")
        exit(0)
    model = load_latest_model('rl_models')
    for i in range(int(sys.argv[2])):
        rate = train(model, 'rl_models', int(sys.argv[3]), int(sys.argv[4]))
        print("Win rate: {}%, Loss rate: {}%, Draw rate: {}%".format(round(rate[0]*100, 1), round(rate[1]*100, 1), round(rate[2]*100, 1)))
else:
    if len(sys.argv) != 4:
        print("Not enough arguments!")
        exit(0)
    model = load_model('rl_models\\rl_policy_{}.keras'.format(sys.argv[2]))
    opponent = load_model('rl_models\\rl_policy_{}.keras'.format(sys.argv[3]))
    rate = test(model, opponent, int(sys.argv[4]))
    time.sleep(0.1)
    print("Win rate: {}%, Loss rate: {}%, Draw rate: {}%".format(round(rate[0]*100, 1), round(rate[1]*100, 1), round(rate[2]*100, 1)))
