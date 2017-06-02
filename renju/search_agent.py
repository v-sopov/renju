import logging
import random
import sys
import os
sys.path.append(os.getcwd())

import backend
import renju
import util

from generate import *
from tree_search import *


def choose_move(search):
    pos = search.make_move()
    # logging.debug('position ' + str(pos))
    return util.to_move(pos)


def main():
    logging.basicConfig(filename='search_agent.log', level=logging.DEBUG)
    # logging.debug("Start search_agent backend...")
    # logging.debug('Current working dir: ' + os.getcwd())
    model_path = os.path.dirname(os.path.realpath(__file__)) + '/large_policy_model'
    # logging.debug('Model path: ' + model_path)
    sl_model = load_model(model_path)
    TIME = 10
    search = MonteCarloTreeSearch(sl_model, None, TIME, 0)

    try:
        while True:
            # logging.debug("Wait for game update...")
            game = backend.wait_for_game_update()
            # logging.debug('Board:\n' + str(game.board()))
            if len(game.positions()) > 0:
                # logging.debug('pass move ' + repr(game.positions()[-1]))
                search.pass_move(game.positions()[-1])
            move = choose_move(search)
            backend.move(move)
            # logging.debug('make move: ' + move)
    except:
        logging.debug('Error!', exc_info=True, stack_info=True)


if __name__ == "__main__":
    main()
