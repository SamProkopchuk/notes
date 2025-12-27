'''
A python implementation of policy iteration pseudocode
(using iterative policy evaluation) found on page 75 of
Reinforcement Learning: An Introduction (2nd ed.)

For Reference:

States:
_____________
| 0| 1| 2| 3|
| 4| 5| 6| 7|
| 8| 9|10|11|
|12|13|14|15|
▔▔▔▔▔▔▔▔▔▔▔▔▔
Where 0 is the terminal state.

Actions:

     ^
     |
     1
<-2     0->
     3
     |
     v

Where actions have the following rewards:
 - (-10) if states {1, 2, 3, 8, 9, 10} are entered
 - (-1) otherwise
'''

import numpy as np


class Policy(object):
    '''
    Policy which chooses action with highest action-value
    '''

    def __init__(self, state_values, use_uniform_random=False):
        self.precompute_actions(state_values)
        self.use_uniform_random = use_uniform_random

    def precompute_actions(self, state_values):
        self.state_to_action = {
            s: max(
                range(4),
                key=lambda a: action_value(
                    a,
                    s,
                    state_values)) for s in range(16)}

    def __call__(self, a, s):
        if s == 0:
            return 0
        elif self.use_uniform_random:
            return 1 / 4
        else:
            return 1 if a == self.state_to_action[s] else 0

    def action(self, s):
        if not self.use_uniform_random:
            return self.state_to_action[s]


def next_state(a, s):
    '''
    Since this environment is deterministic with state-change probabilities ∈ {0, 1},
    it's fine to make a function like this.
    For simplicity, assume input is valid
    '''
    r, c = np.divmod(s, 4)
    if (c == 3 and a == 0 or
        r == 0 and a == 1 or
        c == 0 and a == 2 or
            r == 3 and a == 3):
        return s
    else:
        if (a % 2):
            r += -1 if a == 1 else 1
        else:
            c += -1 if a == 2 else 1
        return 4 * r + c


def expected_reward(a, s):
    '''
    Since we essentially know the 4-variable probability function p's dynamics,
    the following code can be as simple as it is...
    '''
    return -10 if next_state(a, s) in {1, 2, 3, 8, 9, 10} else -1


def action_value(a, s, state_values, lmbda=1):
    return expected_reward(a, s) + lmbda * \
        state_values[np.divmod(next_state(a, s), 4)]


def next_state_value(s, pi: Policy, state_values, lmbda=1):
    '''
    Again, we work around using the 4-value probability function p
    as we have complete knowledge of the environment's dynamics.
    '''
    return sum(pi(a, s) * action_value(a, s, state_values, lmbda=lmbda)
               for a in range(4))


def evaluate_policy(pi: Policy, state_values):
    EPSILON = 1e-4
    delta = np.inf

    while delta > EPSILON:
        delta = 0
        for s in range(16):
            v = state_values[np.divmod(s, 4)]
            state_values[np.divmod(s, 4)] = next_state_value(
                s, pi, state_values)
            delta = max(delta, np.abs(v - state_values[np.divmod(s, 4)]))

    return state_values


def main():
    # Also try np.random.rand :)
    state_values = np.zeros((4, 4))
    pi = Policy(state_values, use_uniform_random=True)

    while True:
        state_values = evaluate_policy(pi, state_values)
        pi_next = Policy(state_values)
        if all(pi.action(s) == pi_next.action(s) for s in range(16)):
            break
        pi = pi_next

    print(f'The optimal state-value function is:\n{state_values}')


if __name__ == '__main__':
    main()
