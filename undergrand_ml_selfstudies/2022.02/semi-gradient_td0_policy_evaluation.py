'''
Gradient Monte Carlo Algorithm for Estimating v^hat = v_pi
Gridworld where going to terminal state -> reward of 1
going to any other state -> reward of 0

        0
       / \
       1 2
     / \\ / \
       ...
   127 ... 254
'''

import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

# Exclude the terminal state 8 and walls:
_STATES = [*range(127)]
_TERMINAL_STATES = set(range(127, 255))
_ACTIONS = [False, True]


class URPolicy:
    '''Uniform Random Policy'''

    def __call__(self, s):
        return random.choice(_ACTIONS)


def environment(s, a):
    sprime = 2 * s + (2 if a else 1)
    r = 0 if sprime < 127 else (-1 if sprime <= 190 else 1)
    return sprime, r


def compress_state(s):
    res = np.zeros((128, 1))
    # We always set the first value to 1 for w_0/b aka the constant.
    if s not in _TERMINAL_STATES:
        res[0] = res[s + 1] = 1
    return res


def state_value(s, w):
    return w.T.dot(s)


def visualize_state_values(w):
    x = [*range(127)]
    y = [state_value(compress_state(s), w).item() for s in x]
    plt.plot(x, y)
    plt.show()


def main(trials=10000, alpha=1e-2, lmbda=1):
    pi = URPolicy()
    w = np.zeros((128, 1))

    for _ in tqdm(range(trials)):
        s = random.choice(_STATES)
        while s not in _TERMINAL_STATES:
            sprime, r = environment(s, pi(s))
            x1, x2 = compress_state(s), compress_state(sprime)
            w += alpha * (r * x1 - x1.dot((x1 - lmbda * x2).T).dot(w))
            s = sprime

    print(f'w aquired via regression:\n{w}')
    visualize_state_values(w)


if __name__ == '__main__':
    main()
