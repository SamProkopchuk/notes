#!/usr/bin/env python

'''
An RL approach to the k-bandits problem.
This example uses the near greedy approach.

The accumulated reward over time is then graphed,
depicting the effects of different epsilons.
'''

__author__ = 'Sam Prokopchuk'

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm

# Different probabilities of not chosing current value with highest expectation
EPSILONS = [0.01, 0.1, 0.5]
TIME_STEPS = 100
NUM_BANDITS = 10


def main():
    n = NUM_BANDITS
    rv_means = np.random.rand(n)
    rvs = [norm(rv_means[i], 1) for i in range(n)]

    rewards_over_time = []
    for e in EPSILONS:
        reward = 0
        reward_over_time = np.zeros(TIME_STEPS)
        action_selected_count = np.zeros(n)
        expected_values = np.zeros(n)

        for t in range(TIME_STEPS):
            if np.random.random() < e:
                # Explore
                action = np.random.randint(n)
            else:
                # Exploit
                action = np.argmax(expected_values)

            action_selected_count[action] += 1
            # Take a single sample from the 'action'th rv.
            sample = rvs[action].rvs()

            expected_values[action] += (sample - expected_values[action]
                                        ) / action_selected_count[action]

            reward += sample
            reward_over_time[t] = reward

        rewards_over_time.append(reward_over_time)

    for i, e in enumerate(EPSILONS):
        plt.plot(np.arange(TIME_STEPS), rewards_over_time[i], label=f'{chr(949)} = {e}')
    plt.xlabel('Time Step')
    plt.ylabel('Accumulated Reward')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
