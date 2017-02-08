#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

def is_equal(a, b):
    if np.array_equal(a, b):
        return True
    a = np.rot90(a)
    if np.array_equal(a, b):
        return True
    a = np.rot90(a)
    if np.array_equal(a, b):
        return True
    a = np.rot90(a)
    if np.array_equal(a, b):
        return True
    a = np.rot90(a)
    a = np.fliplr(a)
    if np.array_equal(a, b):
        return True
    a = np.rot90(a)
    if np.array_equal(a, b):
        return True
    a = np.rot90(a)
    if np.array_equal(a, b):
        return True
    a = np.rot90(a)
    if np.array_equal(a, b):
        return True
    return False

def game_result(state):
    for res in [-1, 1]:
        if np.any(np.all(state == res, axis=1)):
            return res
        if np.any(np.all(state == res, axis=0)):
            return res
        if state[0, 0] == res and state[1, 1] == res and state[2, 2] == res:
            return res
        if state[0, 2] == res and state[1, 1] == res and state[2, 0] == res:
            return res
    if np.all(state != 0):
        return 0
    return 2 # not finished

def generate_states():
    global generate_states_return_value
    if 'generate_states_return_value' in globals():
        return generate_states_return_value
    states = []
    actions = {}
    groups = []
    prev = 0
    cur = 1
    end = 1
    states.append(np.zeros((3, 3), dtype='int8'))
    for number in range(0, 9):
        groups.append(end)
        for state in range(prev, cur):
            if game_result(states[state]) != 2:
                continue
            for i in range(states[state].shape[0]):
                for j in range(states[state].shape[1]):
                    if states[state][i, j] == 0:
                        a = states[state].copy()
                        a[i, j] = 1 - 2*(number % 2)
                        unique = True
                        for old_state in range(cur, end):
                            if is_equal(states[old_state], a):
                                new_state = old_state
                                unique = False
                                break
                        if unique == True:
                            new_state = end
                            states.append(a)
                            end += 1
                        act = actions.get(state, [])
                        if new_state not in act:
                            act.append(new_state)
                        actions[state] = act
        prev = cur
        cur = end
    groups = [0] + groups + [len(states)]
    generate_states_return_value = (states, actions, groups)
    return generate_states_return_value


class TicTacToe:
    def __init__(self, player_x_class, player_o_class, alpha):
        self.states, self.actions, self.state_groups = generate_states()
        x_states = []
        o_states = []
        x_actions = {}
        o_actions = {}
        for i in range(0, len(self.state_groups) - 1, 2):
            x_states.extend(list(range(self.state_groups[i], self.state_groups[i+1])))
            o_states.extend(list(range(self.state_groups[i+1], self.state_groups[i+2])))
        for state in x_states:
            x_actions[state] = self.actions.get(state, [])
        for state in o_states:
            o_actions[state] = self.actions.get(state, [])

        self.player_x = player_x_class(x_states, x_actions, 'x', alpha=alpha)
        self.player_o = player_o_class(o_states, o_actions, 'o', alpha=alpha)

    def train(self, games_num, epsilon=0.2, with_epsilon=1, verbose=False, counter=0, graph=False):
        players = [self.player_x, self.player_o]
        x_wins = 0
        o_wins = 0
        draws = 0
        diffs = [[], []]
        players[0].eps = epsilon
        players[1].eps = epsilon
        for game in range(games_num):
            if game > games_num * with_epsilon:
                players[0].eps = 0
                players[1].eps = 0
            if counter > 0:
                if game % counter == 0:
                    print('game #', game, sep='')
            turn = 0
            state = 0
            if verbose:
                # print(self.states[state])
                print(state)
            while game_result(self.states[state]) == 2:
                state = players[turn % 2].choose(state)
                if verbose:
                    # print(self.states[state])
                    print(state)
                if turn >= 1:
                    reward = game_result(self.states[state])
                    if reward == 2:
                        reward = 0
                    diff = players[(turn+1) % 2].reward(state, reward)
                    diffs[(turn + 1) % 2].append(diff)
                turn += 1
            diff = players[(turn+1) % 2].reward(state, reward)
            diffs[(turn+1) % 2].append(diff)
            reward = game_result(self.states[state])
            if reward == 0:
                draws += 1
            elif reward == 1:
                x_wins += 1
            else:
                o_wins += 1

            if verbose:
                print('=======================================')
        if graph:
            plt.subplot(211)
            plt.plot(range(len(diffs[0])), diffs[0], 'r')
            plt.subplot(212)
            plt.plot(range(len(diffs[1])), diffs[1], 'b')
            plt.show()
        return (x_wins, o_wins, draws, games_num)

    def play(self, games_num, computer):
        if computer == 'x':
            human = Human('o')
            players = [self.player_x, human]
        else:
            human = Human('x')
            players = [human, self.player_o]
        for game in range(games_num):
            turn = 0
            state = 0
            while game_result(self.states[state]) == 2:
                print(self.states[state])
                state = players[turn % 2].choose(state)
                turn += 1
            print(self.states[state])
            if game_result(self.states[state]) == 1:
                print("Winner: X")
            elif game_result(self.states[state]) == -1:
                print("Winner: O")
            else:
                print("Draw")
            print('=======================================')
            print('=======================================')

    def change_player(self, player, new_player):
        if player == 'x':
            self.player_x = new_player
        else:
            self.player_o = new_player


class SARSA:
    def __init__(self, states, actions, side, alpha, trainable=True):
        self.trainable = trainable
        self.states = states
        self.actions = actions
        self.eps = 0
        if side == 'x':
            self.side = 1
        else:
            self.side = -1
        self.alpha = alpha
        self.action_values = {}
        for key in actions.keys():
            self.action_values[key] = np.zeros(len(actions[key]))
        self.step = 1

    def proba(self, state):
        if self.eps >= 0.0001:
            self.eps = 1 / self.step ** 0.33
        proba = np.zeros_like(self.action_values[state])
        best_args = self.action_values[state] == self.action_values[state].max()
        proba[best_args] = (1 - self.eps) / np.argwhere(best_args).shape[0]
        proba += self.eps / proba.shape[0]
        return proba


    def choose(self, state):
        self.old_state = state
        self.last_action =  np.random.choice(len(self.actions[state]), p=self.proba(state))
        return self.actions[state][self.last_action]

    def reward(self, new_state, reward):
        if self.trainable == False:
            return
        reward *= self.side
        old_value = self.action_values[self.old_state][self.last_action]
        next_value = 0
        if len(self.actions.get(new_state, [])) > 0:
            new_choice = np.random.choice(len(self.actions[new_state]), p=self.proba(new_state))
            next_value = self.action_values[new_state][new_choice]
        diff = self.alpha * (reward + next_value - old_value)
        self.action_values[self.old_state][self.last_action] += diff
        self.step += 1
        return abs(diff)


class QLearning:
    def __init__(self, states, actions, side, alpha, trainable=True):
        self.trainable = trainable
        self.states = states
        self.actions = actions
        self.eps = 0
        if side == 'x':
            self.side = 1
        else:
            self.side = -1
        self.alpha = alpha
        self.action_values = {}
        for key in actions.keys():
            self.action_values[key] = np.zeros(len(actions[key]))
        self.step = 1

    def proba(self, state):
        if self.eps >= 0.0001:
            self.eps = 1 / self.step ** 0.33
        proba = np.zeros_like(self.action_values[state])
        best_args = self.action_values[state] == self.action_values[state].max()
        proba[best_args] = (1 - self.eps) / np.argwhere(best_args).shape[0]
        proba += self.eps / proba.shape[0]
        return proba


    def choose(self, state):
        self.old_state = state
        self.last_action =  np.random.choice(len(self.actions[state]), p=self.proba(state))
        return self.actions[state][self.last_action]

    def reward(self, new_state, reward):
        if self.trainable == False:
            return
        reward *= self.side
        old_value = self.action_values[self.old_state][self.last_action]
        next_value = 0
        if len(self.actions.get(new_state, [])) > 0:
            next_value = self.action_values[new_state].max()
        diff = self.alpha * (reward + next_value - old_value)
        self.action_values[self.old_state][self.last_action] += diff
        self.step += 1
        return abs(diff)


class Human:
    def __init__(self, side):
        self.states, self.actions = generate_states()[:2]
        if side == 'x':
            self.marker = 1
        else:
            self.marker = -1

    def choose(self, state):
        row, col = map(int, input('row column (from 1 to 3): ').split())
        row -= 1
        col -= 1
        a = self.states[state].copy()
        a[row, col] = self.marker
        new_state = state + 1
        while True:
            if is_equal(a, self.states[new_state]):
                break
            new_state += 1
        return new_state

    def reward(self, new_state, reward):
        pass

import pickle
with open('states.data', 'rb') as f:
    generate_states_return_value = pickle.load(f)

game = TicTacToe(SARSA, SARSA, 0.2)
res = game.train(10000, epsilon=0.2, with_epsilon=0.95, verbose=False, counter=1000, graph=False)
res = game.train(1000, epsilon=0, with_epsilon=1, verbose=False, counter=0, graph=False)
print(res[2]/res[3])
print('=====')
game = TicTacToe(QLearning, QLearning, 0.1)
res = game.train(10000, epsilon=0.2, with_epsilon=0.95, verbose=False, counter=1000, graph=False)
res = game.train(1000, epsilon=0, with_epsilon=1, verbose=False, counter=0, graph=False)
print(res[2]/res[3])

# game.play(4, computer='x')
# game.play(4, computer='o')