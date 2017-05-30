#!/usr/bin/python3

from generate import *
import sys
from tqdm import tqdm


def idx_to_word(pos):
    height = pos[1] + 1
    width = pos[0] + ord('a')
    if width >= ord('i'):
        width += 1
    width = chr(width)
    return str(width) + str(height)


def word_to_idx(pos):
    width = ord(pos[0])
    if width >= ord('i'):
        width -= 1
    width -= ord('a')
    height = int(pos[1:]) - 1
    return height, width


if len(sys.argv) <= 1:
    print("Usage: in_file [out_file]")
    print("Prints to out.txt by default")
    exit(1)
in_file = sys.argv[1]
out_file = sys.argv[2] if len(sys.argv) >= 3 else "out.txt"
total = 0
for line in open(in_file, 'r'):
    total += 1

out_games = []
results_dict = {0: 'draw', -1: 'black', 1: 'white', 2: 'unknown'}
for game in tqdm(generate_games(sys.argv[1]), total=total):
    result = results_dict[game.result]
    board = np.zeros((15, 15), dtype='float')
    turns = []
    top = 500
    bottom = -500
    left = 500
    right = -500
    for i in range(len(game.turns)):
        height, width = word_to_idx(game.turns[i])
        top = min(top, height)
        bottom = max(bottom, height)
        left = min(left, width)
        right = max(right, width)
        turns.append((width, height))
        board[height, width] = -1 + 2 * (i % 2)
    new_game = result + ' ' + ' '.join((idx_to_word(turn) for turn in turns))
    out_games.append(new_game + '\n')
    for i in range(8):
        vshift = np.random.randint(-top, 15-bottom)
        hshift = np.random.randint(-left, 15-right)
        aug_turns = (idx_to_word((turn[0]+hshift, turn[1]+vshift)) for turn in turns)
        out_games.append(result + ' ' + ' '.join(aug_turns) + '\n')
    # if top > 0:
    #     new_game = result + ' ' + ' '.join((idx_to_word((turn[0], turn[1]-top)) for turn in turns))
    #     out_games.append(new_game + '\n')
    # if bottom < 14:
    #     new_game = result + ' ' + ' '.join((idx_to_word((turn[0], turn[1]+14-bottom)) for turn in turns))
    #     out_games.append(new_game + '\n')
    # if left > 0:
    #     new_game = result + ' ' + ' '.join((idx_to_word((turn[0]-left, turn[1])) for turn in turns))
    #     out_games.append(new_game + '\n')
    # if right < 14:
    #     new_game = result + ' ' + ' '.join((idx_to_word((turn[0]+14-right, turn[1])) for turn in turns))
    #     out_games.append(new_game + '\n')

with open(out_file, 'w') as out:
    for game in out_games:
        out.write(game)



