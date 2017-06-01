import numpy as np
from tqdm import *

def get_rollout_result(board, last_action):  # kind of optimized
    if last_action is None or last_action[0] is None or last_action[1] is None:
        return None
    row, col = last_action[0], last_action[1]
    val = board[row, col]
    if val == 0.0:
        return None
    counts = 1
    for new_row in range(row + 1, 15):
        if board[new_row, col] == val:
            counts += 1
        else:
            break
    for new_row in range(row - 1, -1, -1):
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
    for new_row, new_col in zip(range(row + 1, 15), range(col + 1, 15)):
        if board[new_row, new_col] == val:
            counts += 1
        else:
            break
    for new_row, new_col in zip(range(row - 1, -1, -1), range(col - 1, -1, -1)):
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



def filter_games(in_path, out_path, total=None):
    with open(in_path, 'r') as in_file:
        with open(out_path, 'w') as out_file:
            for line in tqdm(in_file, total=total):
                game = Game(line, out_file)
                game.filter()



class Game:
    def __init__(self, string, file):
        self.initial_string = string
        self.file = file
        if string.startswith("draw"):
            self.result = 0.0
            string = string[5:]
        elif string.startswith("black"):
            self.result = -1.0
            string = string[6:]
        elif string.startswith("white"):
            self.result = 1.0
            string = string[6:]
        elif string.startswith("unknown"):
            self.result = None
            string = string[8:]
        string = string[:-1]
        self.turns = string.split()

    def filter(self):
        board = np.zeros((15, 15), dtype='float')
        width = None
        height = None
        for i in range(len(self.turns)):
            width = ord(self.turns[i][0])
            if width > ord('i'):
                width -= 1
            width -= ord('a')
            height = int(self.turns[i][1:]) - 1
            is_black = (i % 2 == 0)
            board[height, width] = -1 + 2 * (i % 2)
        result = get_rollout_result(board, (height, width))
        if get_rollout_result(board, (height, width)) == self.result and result is not None:
            self.file.write(self.initial_string)


filter_games('test.renju', 'filtered_test.renju', 100000)
