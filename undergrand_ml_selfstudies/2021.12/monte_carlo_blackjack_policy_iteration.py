'''
Monte carlo policy iteration for blackjack

Dealer:
 - While sum(cards) < 17: hit

Agent's policy:
 - To be learned.

Note: For everyone: if sum(cards) <= 11 hit by default.

States:
 - Whether the player has a usable ace (False, True)
 - The sum of the player's cards (12, 21)
  - This is because if sum(cards) < 12 then there's
    0 risk of hitting so we say the agent will hit.
 - The card the dealer shows (1 (Ace), 10)

Actions:
 - Stand: False
 - Hit: True
'''

import numpy as np
import matplotlib.pyplot as plt
from random import randint, choice
from itertools import product

_STATES = [(b, p, d) for b, p, d in product(
    (False, True),
    range(12, 22),
    range(1, 11))]
_ACTIONS = [False, True]


class Policy(object):
    def __init__(self):
        self.policy = {s: choice(_ACTIONS) for s in _STATES}

    def __getitem__(self, s):
        return self.policy[s]

    def __setitem__(self, s, a):
        self.policy[s] = a

    def __call__(self, s):
        return self.policy[s]


def card() -> int:
    return np.clip(randint(1, 14), 1, 10)


def has_usable_ace(cards) -> bool:
    return 1 in cards and sum(cards) + 10 <= 21


def cardsum(cards) -> int:
    return sum(cards) + 10 * has_usable_ace(cards)


def cards_to_state(player_cards, dealer_cards) -> tuple:
    return (
        has_usable_ace(player_cards),
        cardsum(player_cards),
        dealer_cards[0])


def initial_state_to_cards(s) -> tuple:
    # Generate player and dealer cards that would produce an initial state s
    player_cards = [1, s[1] - 11] if s[0] else [s[1] // 2, (s[1] + 1) // 2]
    dealer_cards = [s[2], card()]
    return player_cards, dealer_cards


def generate_episode(s, a, pi) -> int:
    '''
    Generate an episode starting with state s and action a
    '''
    player_cards, dealer_cards = initial_state_to_cards(s)
    res = []
    # sar = "state" "action" "reward"
    sar = [s, a]
    while sar[1]:
        player_cards.append(card())
        sar.append(-1 if cardsum(player_cards) > 21 else 0)
        res.append(sar)
        if cardsum(player_cards) > 21:
            return res
        sar = [cards_to_state(player_cards, dealer_cards),
               pi[cards_to_state(player_cards, dealer_cards)]]
    while cardsum(dealer_cards) < 17:
        dealer_cards.append(card())
    if cardsum(dealer_cards) > 21:
        sar.append(1)
    else:
        ps, ds = cardsum(player_cards), cardsum(dealer_cards)
        sar.append(-1 if ps < ds else (0 if ps == ds else 1))
    res.append(sar)
    return res


def visualize_policy(pi):
    state_to_action = np.zeros((2, 10, 10))
    for b, p, d in product(range(2), range(12, 22), range(1, 11)):
        state_to_action[b][p - 12][d - 1] = int(pi[(bool(b), p, d)])
    for b in range(2):
        plt.imshow(state_to_action[b])
        plt.show()


def main(trials=100000, lmbda=1):
    pi = Policy()
    action_values = {(s, a): 0 for s, a in product(_STATES, _ACTIONS)}
    action_trials = {(s, a): 0 for s, a in product(_STATES, _ACTIONS)}

    for _ in range(trials):
        s = choice(_STATES)
        a = choice(_ACTIONS)
        episode = generate_episode(s, a, pi)
        G = 0
        for s, a, r in episode[::-1]:
            G = lmbda * G + r
            action_values[(s, a)] = (action_values[(s, a)] *
                                     action_trials[(s, a)] + G) / (action_trials[(s, a)] + 1)
            action_trials[(s, a)] += 1
            pi[s] = max((False, True), key=lambda a: action_values[(s, a)])

    visualize_policy(pi)


if __name__ == '__main__':
    main()
