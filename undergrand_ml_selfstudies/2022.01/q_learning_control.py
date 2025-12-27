'''
Q-learning control algorithm for blackjack

Dealer:
 - A deterministic agent which does the following:
 - While sum(cards) < 17: hit

States:
 - Whether the player has a usable ace (False, True)
 - The sum of the player's cards (12, 21)
  - This is because if sum(cards) < 12 then there's
    0 risk of hitting so we say the agent will hit.
 - The card the dealer shows (1 (Ace), 10)
 - Special state: None -> The terminal state

Actions:
 - Stand: False
 - Hit: True
'''

# Use random module since it's like 10x faster
# than numpy.random when vectorization isn't required
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

_STATES = [(b, p, d) for b, p, d in product(
    (False, True),
    range(12, 22),
    range(1, 11))]
_ACTIONS = [False, True]


class Policy(object):
    '''
    Selects argmax(actions) | state given Q function (action values)
    '''

    def __init__(self, action_values):
        # Set policy[state] = argmax(action that provides most reward)
        self.policy = {
            s: max(_ACTIONS, key=lambda a: action_values[s, a]) for s in _STATES}

    def __getitem__(self, s):
        return self.policy[s]

    def __setitem__(self, s, a):
        self.policy[s] = a

    def __call__(self, s):
        return self.policy[s]


def card() -> int:
    return min(10, random.randint(1, 14))


def has_usable_ace(cards) -> bool:
    return 1 in cards and sum(cards) + 10 <= 21


def cardsum(cards) -> int:
    return sum(cards) + 10 * has_usable_ace(cards)


def cards_to_state(player_cards, dealer_cards) -> tuple:
    ps = cardsum(player_cards)
    if ps > 21:
        return None
    else:
        return (has_usable_ace(player_cards), ps, dealer_cards[0])


def state_to_cards(s) -> tuple:
    # Generate player and dealer cards that would produce an initial state s
    player_cards = [1, s[1] - 11] if s[0] else [s[1] // 2, (s[1] + 1) // 2]
    dealer_cards = [s[2], card()]
    return player_cards, dealer_cards


def environment(s, a) -> tuple:
    # (s, a) => (r, s')
    p, d = state_to_cards(s)
    if a:
        p.append(card())
        sprime = cards_to_state(p, d)
        r = 0 if sprime else -1
    else:
        # The only place the dealer's non-face-up cards matter.
        while cardsum(d) < 17:
            d.append(card())
        ps, ds = cardsum(p), cardsum(d)
        sprime = None
        r = -1 if ps < ds and ds <= 21 else (0 if ps == ds else 1)
    return r, sprime


def visualize_policy(pi):
    state_to_action = np.zeros((2, 10, 10))
    for b, p, d in product(range(2), range(12, 22), range(1, 11)):
        state_to_action[b][p - 12][d - 1] = int(pi[bool(b), p, d])
    for b in range(2):
        plt.imshow(state_to_action[b])
        plt.show()


def main(trials=700000, lmbda=1, alpha=5e-3, epsilon=1e-1):
    action_values = {(s, a): 1 for s, a in product(_STATES, _ACTIONS)}
    action_values[None, False] = action_values[None, True] = 0
    # Exaggerate initial expected action-reward to help exploration

    for _ in tqdm(range(trials)):
        s = random.choice(_STATES)
        while s:
            a = max(_ACTIONS, key=lambda x: action_values[s, x])
            if random.random() < epsilon:
                a = not a
            r, sprime = environment(s, a)
            # Now we have s, a, r, sprime
            action_values[s, a] += alpha * (r + lmbda * max(action_values[sprime, _a] for _a in _ACTIONS) - action_values[s, a])
            s = sprime

    policy = Policy(action_values)
    visualize_policy(policy)


if __name__ == '__main__':
    main()
