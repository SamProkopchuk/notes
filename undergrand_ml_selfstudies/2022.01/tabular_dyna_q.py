'''
Tabular Dyna-Q

Gridworld where going to terminal state -> reward of 1
going to any other state -> reward of 0

Actions:

     ^
     |
     1
<-2     0->
     3
     |
     v

#######XG
##X####X#
S#X####X#
##X######
#####X###
#########
'''

import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import product
from tqdm import tqdm
# Efficiency ez clap:
from bisect import bisect_left, insort

# Exclude the terminal state 8 and walls:
_STATES = list(set(range(6 * 9)) - {7, 8, 16, 25, 11, 20, 29, 41})
_ACTIONS = [*range(4)]


def environment(s, a) -> int:
    # Imagine making a readable function WeirdChamp
    if a == 0:
        return (
            s if s %
            9 == 8 or s in {
                10, 19, 28, 6, 15, 24, 40} else s + 1), 0
    elif a == 1:
        return (s if s < 9 or s in {34, 38, 50}
                else s - 9), (1 if s == 17 else 0)
    elif a == 2:
        return (
            s if s %
            9 == 0 or s in {
                8, 12, 17, 21, 26, 30, 42} else s - 1), 0
    else:  # a == 3
        return (s if s > 44 or s in {2, 32} else s + 9), 0


def visualize_action_values(action_values):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for r, c in product(range(6), range(9)):
        s = 9 * r + c
        if s in _STATES:
            uv = {0: [1, 0], 1: [0, 1], 2: [-1, 0], 3: [0, -1]}[max(_ACTIONS, key=lambda a: action_values[s, a])]
            ax.quiver(c+0.5, -r-0.5, *uv)
        elif s == 8:
            rect = Rectangle([c, -r], 1, -1, color='green')
            ax.add_patch(rect)
        else:
            rect = Rectangle([c, -r], 1, -1, color='black')
            ax.add_patch(rect)

    plt.show()

def insort_if_not_contains(l, x):
    i = bisect_left(l, x)
    if i == len(l) or l[i] != x:
        insort(l, x)


def main(trials=5000, lmbda=0.95, alpha=1e-1, epsilon=1e-1, n=20):
    action_to_not = {a: [x for x in _ACTIONS if x != a] for a in _ACTIONS}
    action_values = {(s, a): -1 for s, a in product(_STATES, _ACTIONS)} | {(8, a): 0 for a in _ACTIONS}
    model = {(s, a): None for s, a in product(_STATES, _ACTIONS)}
    seen_s = []

    for _ in tqdm(range(trials)):
        s = random.choice(_STATES)
        a = max(_ACTIONS, key=lambda x: action_values[s, x])
        if random.random() < epsilon:
            a = random.choice(action_to_not[a])

        sprime, r = environment(s, a)
        action_values[s,
                      a] += alpha * (r + lmbda * max(action_values[sprime,
                                                                   _a] for _a in _ACTIONS) - action_values[s,
                                                                                                           a])
        model[s, a] = (sprime, r)
        insort_if_not_contains(seen_s, s)

        for __ in range(n):
            s = random.choice(seen_s)
            a = random.choice([a for a in _ACTIONS if model[s, a] is not None])

            sprime, r = model[s, a]
            action_values[s,
                          a] += alpha * (r + lmbda * max(action_values[sprime,
                                                                       _a] for _a in _ACTIONS) - action_values[s,
                                                                                                               a])
    visualize_action_values(action_values)


if __name__ == '__main__':
    main()
