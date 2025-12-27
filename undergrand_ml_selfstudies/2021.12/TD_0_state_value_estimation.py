'''
TD(0) for estimating v_pi

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
    def __init__(self):
        # By default we just "Stand" in any situation, any other policy can be used.
        self.policy = {s: [1, 0] for s in _STATES}

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


def visualize_state_values(state_values):
    visual_state_values = np.zeros((2, 10, 10))
    for b, p, d in product(range(2), range(12, 22), range(1, 11)):
        visual_state_values[b][p - 12][d - 1] = state_values[(bool(b), p, d)]
    for b in range(2):
        plt.imshow(visual_state_values[b])
        plt.show()


def main(trials=700000, lmbda=1, alpha=0.05):
    pi = Policy()
    state_values = {s: 0 for s in _STATES}
    # The terminal state:
    state_values[None] = 0

    for _ in tqdm(range(trials)):
        s = random.choice(_STATES)
        while s:
            a = random.choices(_ACTIONS, weights=pi(s))[0]
            r, sprime = environment(s, a)
            state_values[s] += alpha * \
                (r + lmbda * state_values[sprime] - state_values[s])
            s = sprime

    visualize_state_values(state_values)


if __name__ == '__main__':
    main()
