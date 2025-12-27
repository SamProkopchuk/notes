#!/usr/bin/env python

'''
An RL approach to a non-stationary k-bandits problem.
This example uses the near greedy approach with
a weighted average value-function.

The accumulated reward over time is then graphed,
depicting the effects of different epsilons.
'''

__author__ = 'Sam Prokopchuk'

import numpy as np
import matplotlib.pyplot as plt


# Feel free to (conservatively) tinker with these constants:

# Bandit 'variables':
# Note: A uniform dist from MIN_MEAN to MAX_MEAN
# will be used to determine the bandits' initial averages.
NUM_BANDITS = 10
MIN_MEAN = -10.
MAX_MEAN = 10.

# Variance of the bandits' reward:
VARIANCE = 1

# Probability of each bandits' mean 'walking' each timestep:
WALK_PROB = 0.1


def walk():
    '''
    Returns value determining how far a bandit will walk
    from its current mean on the offchance of WALK_PROB:
    '''
    return np.random.normal(0, 1)


# Different probabilities of exploring every iteration.
# (Instead of exploiting: choosing current bandit with highest expectation).
EPSILONS = [0.01, 0.1, 0.5, 0.9, 0.99]
# The weight to give the most recent reward signal, (weighted average):
STEP_SIZE = 0.2
TIME_STEPS = 1000


def use_bandit(mean, variance=VARIANCE):
    '''
    Returns reward of a bandit with given mean and variance.
    '''
    return np.random.normal(mean, variance)


def main():
    n = NUM_BANDITS
    bandit_means = np.random.uniform(low=MIN_MEAN, high=MAX_MEAN, size=n)

    # Store the rewards over time for each epsilon in a single ndarray:
    reward_history = np.zeros((len(EPSILONS), TIME_STEPS))
    # Expected values of each bandit for each EPSILON:
    expected_values = np.zeros((len(EPSILONS), n))

    for t in range(TIME_STEPS):
        r = np.random.random()
        for e_idx, e in enumerate(EPSILONS):
            if r < e:
                # Explore
                bandit_idx = np.random.randint(n)
            else:
                # Exploit
                bandit_idx = np.argmax(expected_values[e_idx])

            reward = use_bandit(bandit_means[bandit_idx])
            # Use weighted average to update expectation of used bandit:
            expected_values[e_idx][bandit_idx] += (
                STEP_SIZE * (reward - expected_values[e_idx][bandit_idx]))

            if t == 0:
                reward_history[e_idx][t] = reward
            else:
                reward_history[e_idx][t] = reward_history[e_idx][t-1] + reward
        # Determine which bandits' means should 'walk'
        update_mean_idx = np.argwhere(np.random.random(n) < WALK_PROB)
        # No clean way (afaik) of vectorizing. Also fast enough as is..
        for idx in update_mean_idx:
            bandit_means[idx] += walk()

    for idx, e in enumerate(EPSILONS):
        plt.plot(np.arange(TIME_STEPS), reward_history[idx],
                 label=f'{chr(949)} = {e}')
    plt.xlabel('Time Steps')
    plt.ylabel('Accumulated Reward')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
