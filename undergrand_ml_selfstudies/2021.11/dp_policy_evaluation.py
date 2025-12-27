'''
A python implementation of policy evaluation pseudocode found on page 75 of
Reinforcement Learning: An Introduction (2nd ed.)

For Reference:

States:
_____________
| 0| 1| 2| 3|
| 4| 5| 6| 7|
| 8| 9|10|11|
|12|13|14|15|
▔▔▔▔▔▔▔▔▔▔▔▔▔
Where 0, 15 are terminal states.

Actions:

     ^
     |
     1
<-2     0->
     3
     |
     v

Where all actions have reward of -1 (until terminal state reached)
'''

import numpy as np


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


def policy(a, s):
    '''
    If s is a terminal state then return 0
    since you can't select actions in terminal state
    '''
    return 0 if s in (0, 15) else 0.25


def expected_reward(a, s):
    '''
    Since we essentially know the 4-variable probability function p's dynamics,
    the following code can be as simple as it is...
    '''
    return -1


def calculate_next_state_value(s, current_state_values, lmbda=1):
    '''
    Again, we work around using the 4-value probability function p
    as we have complete knowledge of the environment's dynamics.
    '''
    res = 0
    for a in range(4):
        res += policy(a, s) * (expected_reward(a, s) + lmbda *
                               current_state_values[np.divmod(next_state(a, s), 4)])
    return res


def main():
    # Also try np.random.rand :)
    current_state_values = np.zeros((4, 4))
    next_state_values = current_state_values.copy()
    # Note it's possible to do this in-place
    # (According to textbook it's often more-efficient too)

    EPSILON = 1e-4
    delta = np.inf

    while delta > EPSILON:
        for s in range(16):
            next_state_values[np.divmod(s, 4)] = calculate_next_state_value(
                s, current_state_values)
        delta = np.abs(next_state_values - current_state_values).max()
        current_state_values[::] = next_state_values

    print(current_state_values)


if __name__ == '__main__':
    main()
