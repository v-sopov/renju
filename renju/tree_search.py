import numpy as np
from generate import prepare_board, show_board, legal_nn_move
import sys
from tqdm import *
import time
import mcts_utils
import math
from itertools import chain

NB_CANDIDATES = 5
NB_ROLLOUTS = 1
NB_ROLLOUT_NN = 0
NB_ROLLOUT_FAST = 0

NODE_COUNTER = 1
symbols = [chr(x) for x in chain(range(ord('a'), ord('z')+1),
                                 range(ord('A'), ord('Z')+1))]

def id_to_str(id_num):
    global symbols
    length = len(symbols)
    letters = []
    while id_num != 0:
        res = divmod(id_num, length)
        letters.append(symbols[res[1]])
        id_num = res[0]
    return ''.join(letters)


def idx_to_word(pos):
    height = pos[0] + 1
    width = pos[1] + ord('a')
    if width >= ord('i'):
        width += 1
    width = chr(width)
    return str(width) + str(height)

def get_result(board):  # TODO: optimize it. heavily
    for i in range(2):
        for row in range(board.shape[0]):
            for col in range(board.shape[1] - 4):
                black = True
                white = True
                for i in range(5):
                    if board[row, col+i] != 1.0:
                        white = False
                    if board[row, col+i] != -1.0:
                        black = False
                if black:
                    return -1.0
                if white:
                    return 1.0
        board = np.rot90(board)
    for i in range(2):
        for row in range(board.shape[0] - 4):
            for col in range(board.shape[1] - 4):
                black = True
                white = True
                for i in range(5):
                    if board[row+i, col+i] != 1.0:
                        white = False
                    if board[row+i, col+i] != -1.0:
                        black = False
                if black:
                    return -1.0
                if white:
                    return 1.0
        board = np.rot90(board)
    if np.abs(board).sum() == board.size:
        return 0.0
    else:
        return None


def legal_move(board, policy):
    is_black = (int(board.sum()) == 0)
    p_board = prepare_board(board,
                            is_black)
    p_board = p_board.flatten()
    probas = policy.predict_proba(p_board[np.newaxis])[0]
    mask = 1.0 - np.abs(board).flatten()
    probas *= mask
    res = np.argmax(probas)
    return res // 15, res % 15


class TreeNode:
    def __init__(self, board, action, is_black_to_go, sl_policy):
        global NB_CANDIDATES
        self.sl_policy = sl_policy
        self.children = {}
        self.board = board.copy()
        if action is not None:
            self.board[action] = -1 - 2*self.board.sum()
        self.values_sum = 0.0
        self.visits = 0
        self.prior_proba = 1.0
        self.last_expanded_action = -1
        self.is_black_to_go = is_black_to_go
        self.children_probas = self.sl_policy.predict(prepare_board(self.board, not self.is_black_to_go)[np.newaxis])[0]
        mask = 1.0 - np.abs(self.board).flatten()
        self.candidates = [x for x in
                           np.argpartition(mask * (self.children_probas + 0.00001), 224-NB_CANDIDATES)[-NB_CANDIDATES:]]
        self.candidates.sort(key=lambda x: self.children_probas[x], reverse=True)
        self.candidates = [(x // 15, x % 15) for x in self.candidates]
        self.max_children = min(round(self.board.size - np.abs(self.board).sum()), len(self.candidates))
        self.expand_times = 0
        global NODE_COUNTER
        self.graph_name = id_to_str(NODE_COUNTER)
        self.graph_label = idx_to_word(action) if action is not None else '--'
        NODE_COUNTER += 1

    def expand(self):
        self.expand_times += 1
        action = self.last_expanded_action + 1
        while True:
            if self.board[self.candidates[action]] == 0.0:
                break
            else:
                action += 1
        self.last_expanded_action = action
        new_node = TreeNode(self.board, self.candidates[action], not self.is_black_to_go, self.sl_policy)
        new_node.prior_proba = self.children_probas[15*self.candidates[action][0] + self.candidates[action][1]]
        self.children[self.candidates[action]] = new_node
        return new_node

    def get_interest_value(self, final):
        res = self.get_q_value()
        if not final:
            res += (2 + 2*self.prior_proba) / math.sqrt(1.0 + self.visits)
        return res

    def get_q_value(self):
        return self.values_sum / max(1, self.visits)

    def get_best_action(self, final=False):
        max_action = None
        max_interest = None
        for action in self.children:
            new_interest = self.children[action].get_interest_value(final)
            if max_action is None or new_interest > max_interest:
                max_action = action
                max_interest = new_interest
        return max_action

    def generate_graph(self, out, depth=0, root=None):
        if depth == 0:
            root = self
        result = get_result(self.board)
        if result is not None:
            if result == (-1) and not root.is_black_to_go or result == 1 and root.is_black_to_go:
                color = "#88FF88"
            elif result == 1 and not root.is_black_to_go or result == (-1) and root.is_black_to_go:
                color = "#FF8888"
            else:
                color = "#FFFF66"
        else:
            color = "#DDDDDD"
        val = round(self.get_interest_value(depth == 1), 1)
        print('{} [label="{}\\n{}\\n{}",color="{}",style=filled]'.format(self.graph_name,
                                                      self.graph_label,
                                                      val,
                                                      self.visits, color),
                                                                  file=out)
        for child in self.children:
            self.children[child].generate_graph(out, depth+1, root)
            print('    "{}" -> "{}"'.format(self.graph_name, self.children[child].graph_name), file=out)


class MonteCarloTreeSearch:
    def __init__(self, sl_policy, rollout_policy, time, verbose=1):
        self.verbose = verbose
        self.node_list = []
        root_board = np.zeros((15, 15), dtype='float32')
        self.root = TreeNode(root_board, None, False, sl_policy)
        self.sl_policy = sl_policy
        self.rollout_policy = rollout_policy
        self.time = time

    def make_move(self):
        start_time = time.clock_gettime(time.CLOCK_REALTIME)
        end_time = start_time + self.time - 0.1
        checked_action = mcts_utils.find_fours(self.root.board, not self.root.is_black_to_go)
        if checked_action is not None:
            self.pass_move(checked_action)
            return checked_action
        if self.verbose == 2:
            show_board(self.root.children_probas.reshape((15, 15)))
        if self.verbose != 0:
            progress_bar = trange(self.time*30)
            progress = start_time*30
            iteration = 1
        while time.clock_gettime(time.CLOCK_REALTIME) < end_time:
            if self.verbose != 0:
                increment = int(30*time.clock_gettime(time.CLOCK_REALTIME) - progress)
                if increment > 0:
                    progress_bar.set_postfix(iter=str(iteration))
                    progress_bar.update(increment)
                    progress += increment
            if self.verbose == 2 and end_time - time.clock_gettime(time.CLOCK_REALTIME) < 0.01:
                with open('graph.txt', 'w') as f:
                    print('digraph {', file=f)
                    self.root.generate_graph(f)
                    print('}', file=f)
            node = self.select(self.root)
            reward = self.rollout(node)
            self.backpropagate(reward)
            self.node_list = []
            if self.verbose != 0:
                iteration += 1
        if self.verbose != 0:
            progress_bar.update(1000)
            progress_bar.close()
        action = self.root.get_best_action(final=True)
        self.root = self.root.children[action]
        return action

    def pass_move(self, action):
        if action not in self.root.children:
            new_node = TreeNode(self.root.board, action, not self.root.is_black_to_go, self.sl_policy)
            self.root.children[action] = new_node
        self.root = self.root.children[action]

    def select(self, node):
        action = (0, 0)
        while mcts_utils.get_rollout_result(node.board, action) is None:
            self.node_list.append(node)
            if node.expand_times < node.max_children:
                new_node = node.expand()
                self.node_list.append(new_node)
                return new_node
            else:
                action = node.get_best_action()
                node = node.children[action]
        self.node_list.append(node)
        return node

    def rollout(self, node):
        global NB_ROLLOUTS
        global NB_ROLLOUT_NN
        global NB_ROLLOUT_FAST
        board = node.board.copy()
        action = None
        counter = 0
        while mcts_utils.get_rollout_result(board, action) is None:
            if counter < NB_ROLLOUT_NN:
                action = legal_nn_move(board, self.sl_policy)
            elif counter < NB_ROLLOUT_NN + NB_ROLLOUT_FAST:
                action = legal_move(board, self.rollout_policy)
            else:
                break
            board[action] = -1.0 - 2*board.sum()
            counter += 1
        result = mcts_utils.get_rollout_result(board, action)
        if result is not None:
            return result
        result = 0.0
        is_black_saved = (board.sum() == 0.0)
        for i in range(NB_ROLLOUTS):
            middle_board = board.copy()
            is_black = is_black_saved
            while mcts_utils.get_rollout_result(middle_board, action) is None:
                action = mcts_utils.full_near(middle_board, is_black)
                middle_board[action] = -1.0 - 2*middle_board.sum()
                is_black = not is_black
            result += mcts_utils.get_rollout_result(middle_board, action)
        result /= NB_ROLLOUTS
        return result

    def backpropagate(self, reward):
        for node in self.node_list:
            node.visits += 1
            fixed_reward = -reward if node.is_black_to_go else reward
            node.values_sum += fixed_reward
