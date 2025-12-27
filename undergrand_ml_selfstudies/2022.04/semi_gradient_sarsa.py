'''
The moutain car problem
Solved via semi-gradient sarsa using linear regression with tiling
'''
import random
import numpy as np

from math import cos
from tile import tile_encode
from tqdm import tqdm

import matplotlib.pyplot as plt

# Params you can f with:

_EPSILON = 1e-10

# FULL THROTTLE REVERSE: -1
# ZERO THROTTLE: 0
# FULL THROTTLE FORWARD: 1
_ACTIONS = [-1, 0, 1]

_POS_BOUNDS = (-1.2, 0.5)
_VEL_BOUNDS = (-0.07, 0.07)

_POS_BUCKETS = 8
_VEL_BUCKETS = 8

_N_TILINGS = 8
_STATE_SHAPE = (len(_ACTIONS) * _POS_BUCKETS * _VEL_BUCKETS * _N_TILINGS, )
_CANVAS_SHAPE = (3 * _N_TILINGS, _POS_BUCKETS * _VEL_BUCKETS)


def clip(a, a_min, a_max):
    '''More efficient than np.clip cuz doesn't need to be a ufunc'''
    return min(a_max, max(a_min, a))


def next_state(s, a):
    curr_pos, curr_vel = s
    next_vel = clip(curr_vel + 0.001 * a - 0.0025 *
                    cos(3 * curr_pos), *_VEL_BOUNDS)
    next_pos = clip(curr_pos + next_vel, *_POS_BOUNDS)
    if next_pos == _POS_BOUNDS[0]:
        next_vel = 0
    return next_pos, next_vel


def state_action_to_x(s, a):
    te = tile_encode(
        [s],
        [_POS_BOUNDS, _VEL_BOUNDS],
        [_POS_BUCKETS, _VEL_BUCKETS],
        _N_TILINGS).ravel()
    res = np.zeros((te.shape[0] * 3, ))
    res[(a + 1) * te.shape[0]: (a + 2) * te.shape[0]] = te
    return res


def environment(s, a):
    return -1, next_state(s, a)


def state_action_value(x, w):
    return w.T.dot(x)


def best_action(s, w):
    return max(
        _ACTIONS,
        key=lambda a:
        state_action_value(
            state_action_to_x(s, a),
            w))


def is_terminal_state(s):
    return s[0] == _POS_BOUNDS[1]


def graph_state_action_values(w, nrows=128, ncols=128):
    mp = np.zeros((nrows, ncols))
    for row, p in enumerate(np.linspace(*_POS_BOUNDS, nrows, endpoint=True)):
        for col, v in enumerate(np.linspace(*_VEL_BOUNDS, ncols, endpoint=True)):
            mp[row, col] = best_action((p, v), w)
    plt.imshow(
        mp,
        cmap='binary',
        interpolation='nearest',
        origin='lower',
        extent=[*_VEL_BOUNDS, *_POS_BOUNDS],
        aspect=(_VEL_BOUNDS[1] - _VEL_BOUNDS[0]) / (_POS_BOUNDS[1] - _POS_BOUNDS[0]))
    plt.xlabel('Velocity')
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()


def fit(trials=1000, lmbda=1, alpha=0.5 / _N_TILINGS, epsilon=0, debug=False):
    w = np.zeros(_STATE_SHAPE)
    action_to_not = {a: [x for x in _ACTIONS if x != a] for a in _ACTIONS}
    if debug:
        fig, ax = plt.subplots()
        heatmap = ax.pcolor(
            w.reshape(_CANVAS_SHAPE),
            cmap='binary',
            edgecolors='white')
        fig.canvas.draw()
        fig.show()

    for _ in tqdm(range(trials)):
        s = [random.uniform(*_POS_BOUNDS), 0]  # random.uniform(*_VEL_BOUNDS)]
        a = best_action(s, w)
        if random.random() < epsilon:
            a = random.choice(action_to_not[a])
        if not is_terminal_state(s):
            while True:
                r, sprime = environment(s, a)
                x = state_action_to_x(s, a)
                if debug:
                    heatmap = ax.pcolor(
                        w.reshape(_CANVAS_SHAPE),
                        cmap='binary',
                        edgecolors='white')
                    ax.draw_artist(ax.patch)
                    ax.draw_artist(heatmap)
                    fig.canvas.blit(ax.bbox)
                    fig.canvas.flush_events()
                    if s[0] > -0.35:
                        print(f'{s} {a} {sprime} {state_action_value(x, w)} {w.min()} {w.max()}')
                if is_terminal_state(sprime):
                    w += alpha * (r - state_action_value(x, w)) * x
                    break
                aprime = best_action(sprime, w)
                if random.random() < epsilon:
                    aprime = random.choice(action_to_not[aprime])
                xprime = state_action_to_x(sprime, aprime)
                w += alpha * (
                    r +
                    lmbda * state_action_value(xprime, w) -
                    state_action_value(x, w)) * x
                s, a = sprime, aprime
    graph_state_action_values(w)
    return w


def render_episodes(w, nepisodes=10):
    import gym
    env = gym.make('MountainCar-v0')
    # env.metadata['render_fps'] = 5
    for _ in range(nepisodes):
        s = env.reset()
        while True:
            a = best_action(s, w) + 1
            s, r, done, d = env.step(a)
            env.render()
            if done:
                break
    env.close()


def main():
    w = fit()
    render_episodes(w)


if __name__ == '__main__':
    main()
