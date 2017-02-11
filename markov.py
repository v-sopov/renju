
# coding: utf-8

# ## 2. Марковский процесс принятия решения (markov decison process)

# ### Задание
# #### 1. Приведите жизненный пример марковского процесса принятия решения (это может быть какаю-нибудь игра и т.п.).
# Очень простой пример - шахматы. Конечное, пусть и очень большое, количество состояний $S$, конечное множество действий $ A \subset S \times S$. Среда - наш противник - отвечает вполне себе случайно. Если рассматривать среду именно как противника, то она, конечно, отдаёт сильное предпочтение тем действиям, которые она считает выигрышными. Награда для каждого допустимого действия из $\; S \times A \;$ нулевая за исключением последнего хода, по результатам которого присуждается награда $\pm 1$. Кроме того, очевидно, что для принятия решения информация о предыдущих состояниях нерелевантна. Таким образом, принятие решения в шахматах можно смоделировать с помощью марковского процесса.
# #### 2. Можете ли вы привести пример игры, где принятие решения нельзя смоделировать с помощью марковского процесса?
# Крестики-нолики на бесконечной доске не могут быть смоделированы хотя бы из-за бесконечности (в частности, континуальности) множества состояний $S$.
# #### 3. Выведите следующие значения через $p(s_{t+1}, r_{t+1}|s_t, a_t)$, для простоты все распределения можно считать дискретными
#   * $r(s_{t}, a_{t}) = \mathbb{E}[R_{t+1}|S_t = s_t, A_t = a_t]$ - средняя награда за действие $a_t$ в $s_t$ 
#   * $p(s_{t+1} | s_t, a_t) = \Pr\{S_{t+1} = s_{t+1} | S_t = s_t, A_t = a_t \}$ - вероятность попасть в $s_{t+1}$ из $s_t$, сделав $a_t$.
#   * $r(s_t, a_t, s_{t+1}) = \mathbb{E}[R_{t+1}|S_{t+1} = s_{t+1}, S_t = s_t, A_t = a_t]$ - средняя награда при переезде из $s_t$ в $s_{t+1}$, сделав $a_t$.
# 
# $r(s_{t}, a_{t}) = \mathbb{E}[R_{t+1}|S_t = s_t, A_t = a_t] = \sum_{s,r} r \cdot p(s, r \;|\; s_t, a_t)$
# 
# $p(s_{t+1} | s_t, a_t) = \Pr\{S_{t+1} = s_{t+1} | S_t = s_t, A_t = a_t \} = \sum_{r} p(s_{t+1}, r \;|\; s_t, a_t)$
# 
# $r(s_t, a_t, s_{t+1}) = \mathbb{E}[R_{t+1}|S_{t+1} = s_{t+1}, S_t = s_t, A_t = a_t] = \sum_{r} r \cdot p(s_{t+1}, r \;|\; s_t, a_t)$
# 
# 

# ### Смоделируем среду:

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.ndimage.filters import gaussian_filter1d


def log_progress(sequence, every=None, size=None):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{index} / ?'.format(index=index)
                else:
                    progress.value = index
                    label.value = u'{index} / {size}'.format(
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = str(index or '?')


# ### Построим такую модель:
# Неформально определим её так: агент может пытаться двигаться на целочисленном отрезке $\mathbb{N} \cap [0;49]$ вправо или влево. При выборе направления движения агент может случайным образом сдвинуться в соответсвующую сторону на $1$ с вероятностью $\frac{5}{16}$, на $2$ с вероятностью $\frac{4}{16}$, на $3$ с вероятностью $\frac{3}{16}$, на $4$ с вероятностью $\frac{2}{16}$, на $5$ с вероятностью $\frac{1}{16}$. Кроме того, с вероятностью $\frac{1}{16}$ агент может остаться на месте.
# 
# В 40-ом состоянии агент получает награду $10$. В 42-ом он получает штраф $100$. Во всех прочих состояниях он ничего не получает.
# 
# Игра бесконечна.

# In[131]:

class Environment:
    def __init__(self, states, actions=2):
        self.states = states
        self.actions = actions
        self.transition_probas = np.zeros((states, actions, states))
        self.rewards = np.zeros(states)
        self.good_state = states - 10
        self.bad_state = states - 8
        self.rewards[self.good_state] = 10
        self.rewards[self.bad_state] = -100
        for state in range(states):
            for i in range(state+1, min(states, state+6)):
                self.transition_probas[state, 0, i] = 6 - (i - state)
            self.transition_probas[state, 0, state] = 1
            for i in range(state-1, max(-1, state-6), -1):
                self.transition_probas[state, 1, i] = 6 - (state - i)
            self.transition_probas[state][1, state] = 1
            
            self.transition_probas[state, 0] /= self.transition_probas[state, 0].sum()
            self.transition_probas[state, 1] /= self.transition_probas[state, 1].sum()
                
        
    
    def step(self, state, action):
        new_state = np.random.choice(self.states, p=self.transition_probas[state, action])
        return (new_state, self.rewards[new_state])


# In[132]:

class PolicyIterationStrategy:
    def __init__(self, env, discount):
        self.env = env
        self.discount = discount
        self.policy = np.ones((env.states, env.actions))
        self.policy /= env.actions
    def learn(self):
        self.state_values = np.random.randn(self.env.states)
        self.state_action_values = np.zeros((self.env.states, self.env.actions))
        i = 0
        while True:
            while True:
                epsilon = 0
                for state in range(self.env.states):
                    old_value = self.state_values[state]
                    val_sum = 0
                    for action in range(self.env.actions):
                        action_sum = 0
                        for new_state in range(self.env.states):
                            action_sum += self.env.transition_probas[state, action, new_state] *                                 (self.env.rewards[new_state] +                                     self.discount*self.state_values[new_state])
                        val_sum += self.policy[state, action] * action_sum
                    self.state_values[state] = val_sum
                    epsilon = max(epsilon, abs(old_value - self.state_values[state]))
                i += 1
                if epsilon < 0.0001:
                    break
                
            stable = True    
            for state in range(self.env.states):
                comp_values = self.state_action_values[state]
                old_best = np.argwhere(comp_values == comp_values.max())
                for action in range(self.env.actions):
                    action_sum = 0
                    for new_state in range(self.env.states):
                        action_sum += self.env.transition_probas[state, action, new_state] *                             (self.env.rewards[new_state] +                                 self.discount*self.state_values[new_state])
                    self.state_action_values[state, action] = action_sum
                comp_values = self.state_action_values[state]
                best = np.argwhere(comp_values == comp_values.max())
                self.policy[state] = 0
                self.policy[state][best] = 1
                self.policy[state] /= self.policy[state].sum()
                if not np.array_equal(old_best, best):
                    stable = False
            if stable:
                break
        return i
                    
                
    def choose(self, state):
        return np.random.choice(self.env.actions, p=self.policy[state])
    

class ValueIterationStrategy:
    def __init__(self, env, discount):
        self.env = env
        self.discount = discount
        self.policy = np.ones((env.states, env.actions))
        self.policy /= env.actions
    def learn(self):
        self.state_values = np.random.randn(self.env.states)
        self.state_action_values = np.zeros((self.env.states, self.env.actions))
        i = 0
        while True:
            epsilon = 0
            for state in range(self.env.states):
                old_value = self.state_values[state]
                max_val = -1e20
                for action in range(self.env.actions):
                    action_sum = 0
                    for new_state in range(self.env.states):
                        action_sum += self.env.transition_probas[state, action, new_state] *                             (self.env.rewards[new_state] +                                 self.discount*self.state_values[new_state])
                    max_val = max(max_val, action_sum)
                self.state_values[state] = max_val
                epsilon = max(epsilon, abs(old_value - self.state_values[state]))
            i += 1
            if epsilon < 0.0001:
                break
                
        for state in range(self.env.states):
            res = np.zeros(self.env.actions)
            for action in range(self.env.actions):
                action_sum = 0
                for new_state in range(self.env.actions):
                    for new_state in range(self.env.states):
                        action_sum += self.env.transition_probas[state, action, new_state] *                             (self.env.rewards[new_state] +                                 self.discount*self.state_values[new_state])
                res[action] = action_sum
            best = np.argwhere(res == res.max())
            self.policy[state] = 0
            self.policy[state][best] = 1
            self.policy[state] /= self.policy[state].sum()
        
        return i
                
    def choose(self, state):
        return np.random.choice(self.env.actions, p=self.policy[state])
    
class RandomStrategy:
    def __init__(self, env, discount):
        self.env = env
        self.discount = discount
        self.policy = np.ones((env.states, env.actions))
        self.policy /= env.actions
    def learn(self):
        return 0
    def choose(self, state):
        return np.random.choice(self.env.actions, p=self.policy[state])


# In[133]:

class MarkovPlayer:
    def __init__(self, states, actions, steps, strategy_class, discount):
        self.states = states
        self.actions = actions
        self.strategy_class = strategy_class
        self.steps = steps
        self.discount = discount
    def evaluate(self, games=1, progressbar=True,hold=False, show=True,
                 color='b', label=' '):
        rewards = np.zeros(self.steps)
        if progressbar:
            games_range = log_progress(range(games), every=1)
        else:
            games_range = range(games)
        avg_time = 0
        for game in games_range:
            state = 0
            self.state = state
            env = Environment(self.states, self.actions)
            strategy = self.strategy_class(env, self.discount)
            self.strategy = strategy
            avg_time += strategy.learn()
            for i in range(self.steps):
                state, r = env.step(state, strategy.choose(state))
                rewards[i] += r
                self.state = state
        avg_time /= games
        discount = self.discount
        print(avg_time, 'steps to converge')
        for i in range(1, self.steps):
            rewards[i] = rewards[i-1] + discount * rewards[i]
            discount *= self.discount
        for i in range(1, self.steps):
            rewards[i] /= games
        if show:
            x = np.arange(1, self.steps+1)
            plot_rewards = rewards
            plt.plot(x, plot_rewards, color, label=label)
            plt.title('Total reward', fontsize=16)
            if not hold:
                plt.show()
        return rewards.sum()


# ### Сравним эффективность и скорость сходимости алгоритмов:

# ## ATTENTION: Не запускать следующий код с исходным games_num. Долго работает. Для проверки работы достаточно games_num ~ 50 или даже меньше

# In[134]:

get_ipython().run_cell_magic('time', '', "# == IMPORTANT PARAMETER ==\ngames_num = 3000\n# =========================\nsteps = 120\ndiscount = 0.95\nstates = 50\nactions = 2\nrandom_player = MarkovPlayer(states=states, actions=actions, steps=steps,\n                           strategy_class=RandomStrategy, discount=discount)\npolicy_iteration_player = MarkovPlayer(states=states, actions=actions, steps=steps,\n                           strategy_class=PolicyIterationStrategy, discount=discount)\nvalue_iteration_player = MarkovPlayer(states=states, actions=actions, steps=steps,\n                           strategy_class=ValueIterationStrategy, discount=discount)\nplt.hold(True)\nrandom_player.evaluate(games=games_num, hold=True, color='k')\nprint('===')\npolicy_iteration_player.evaluate(games=games_num, hold=True, color='r')\nprint('===')\nvalue_iteration_player.evaluate(games=games_num, hold=True, color='b')\nplt.hold(False)\nplt.show()")


# Как можно заметить, оба алгоритма показывают схожие результаты (что логично, так как они должны находить одну оптимальную стратегию), однако **value iteration strategy** сходится всё же гораздо быстрее.
# 
# ### Теперь посмотрим, как зависит значение $V_\gamma^*(s)$ для пяти произвольных состояний:

# In[153]:

states = 50
actions = 2
env = Environment(states, actions)
gammas = [x*0.01 for x in range(1, 98) if x % 2 == 1]
gammas = gammas + [0.985, 0.986, 0.99]
strategies = []
states = {}
for gamma in log_progress(gammas):
    strategies.append(ValueIterationStrategy(env, gamma))
    strategies[-1].learn()
plt.hold(True)
for state, color in zip([0, 20, 37, 42], ['k', 'r', 'g', 'b']):
    states[state] = [strategy.state_values[state] for strategy in strategies[:-3]]
    plt.plot(gammas[:-3], states[state], color, label='state #' + str(state))
states[49] = [strategy.state_values[49] for strategy in strategies]
plt.plot(gammas, states[49], 'm', label='state #' + str(49))

plt.hold(False)
plt.xlabel('discount rate')
plt.ylabel('V*(s)')
plt.legend(loc='upper left')
plt.show()


# Начальное положение кривых относительно друг друга понятно: при низком значении дисконтирования очень сильно учитывается награда, полученная в самом начале, и почти не оценивается награда, полученная в самом конце. Из-за этого начальная ценность всех состояний, кроме 42-го при попадании в которое начисляется штраф, близка к нулю. Затем при росте $\gamma$ заметна следующая тенденция: чем ближе к "выигрышнному" состоянию мы находимся, тем скорее текущее состояние набирает ценность при росте $\gamma$ в начале, однако затем они более-менее выравниваются. Это происходит из-за того, что более близкие к выигрышному состояния раньше начинают получать выигрыш, и он, соответственно, в меньшей степени подвергается дисконтированию. Потом этот разрыв уменьшается, так как все состояния рано или поздно достигают выигрышного, и начальная фора начинает выражаться просто константой
# 
# Интересным исключением в данной модели является только 49-ое состояния. Так как от выигрышного состояния его отделяет большой штраф, при низких значениях $\gamma$ от просто не успевает скомпенсировать его, в результате чего ему становится выгоднее не рисковать и идти в противоположную сторону, где есть лишь нулевая награда. Однако с определённого момента последующая после штрафа награда уже начинает перевешивать штраф, и происходит взрывной рост ценности

# In[ ]:



