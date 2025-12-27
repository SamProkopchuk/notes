'''
Monte carlo epsilon-soft policy iteration for blackjack

Dealer:
 - A deterministic agent which does the following:
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


class EpsilonSoftPolicy(object):
    def __init__(self):
        self.policy = {s: [1 / len(_ACTIONS)] * len(_ACTIONS) for s in _STATES}

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
    return (
        has_usable_ace(player_cards),
        cardsum(player_cards),
        dealer_cards[0])


def initial_state_to_cards(s) -> tuple:
    # Generate player and dealer cards that would produce an initial state s
    player_cards = [1, s[1] - 11] if s[0] else [s[1] // 2, (s[1] + 1) // 2]
    dealer_cards = [s[2], card()]
    return player_cards, dealer_cards


def generate_episode(s, pi) -> int:
    '''
    Generate an episode starting with state s
    and following epsilon-soft policy pi
    '''
    player_cards, dealer_cards = initial_state_to_cards(s)
    res = []
    # sar = "state" "action" "reward"
    sar = [s, *random.choices(_ACTIONS, weights=pi[s])]
    while sar[1]:
        player_cards.append(card())
        sar.append(-1 if cardsum(player_cards) > 21 else 0)
        res.append(sar)
        if cardsum(player_cards) > 21:
            return res
        s = cards_to_state(player_cards, dealer_cards)
        sar = [s, *random.choices(_ACTIONS, weights=pi[s])]
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
        state_to_action[b][p - 12][d - 1] = pi[(bool(b), p, d)][1]
    for b in range(2):
        plt.imshow(state_to_action[b])
        plt.show()


def main(trials=700000, lmbda=1, epsilon=4e-1):
    pi = EpsilonSoftPolicy()
    action_values = {(s, a): 0 for s, a in product(_STATES, _ACTIONS)}
    action_trials = {(s, a): 0 for s, a in product(_STATES, _ACTIONS)}

    for _ in tqdm(range(trials)):
        s = random.choice(_STATES)
        a = random.choice(_ACTIONS)
        episode = generate_episode(s, pi)
        G = 0
        for s, a, r in episode[::-1]:
            G = lmbda * G + r
            action_values[(s, a)] = (action_values[(s, a)] *
                                     action_trials[(s, a)] + G) / (action_trials[(s, a)] + 1)
            action_trials[(s, a)] += 1
            optimal_action = max(
                (False, True), key=lambda a: action_values[(s, a)])
            pi[s] = [
                (1 -
                 epsilon +
                 epsilon /
                 len(_ACTIONS) if a == optimal_action else epsilon /
                    len(_ACTIONS)) for a in _ACTIONS]

    visualize_policy(pi)


if __name__ == '__main__':
    main()
