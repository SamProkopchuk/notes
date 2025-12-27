'''
State-value monte carlo approximation for blackjack

Dealer:
 - While sum(cards) < 17: hit

Agent's policy:
 - If sum(cards) < 20: hit
   - Otherwise, stand.

States:
 - Whether the player has a usable ace (False, True)
 - The sum of the player's cards (12, 21)
 - The card the dealer shows (1 (Ace), 10)
'''

import numpy as np
import matplotlib.pyplot as plt
from random import randint
from itertools import product

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


def simulate_and_get_reward(player_cards, dealer_cards) -> int:
    '''
    Since γ=1 and we know intermediate actions have no reward.
    We just want the end reward - hence this function.
    '''
    while cardsum(player_cards) < 20:
        player_cards.append(card())
    if cardsum(player_cards) > 21:
        return -1
    while cardsum(dealer_cards) < 17:
        dealer_cards.append(card())
    if cardsum(dealer_cards) > 21:
        return 1
    else:
        ps, ds = cardsum(player_cards), cardsum(dealer_cards)
        return -1 if ps < ds else (0 if ps == ds else 1)


def visualize_state_values(state_values):
    visual_state_values = np.zeros((2, 10, 10))
    for b, p, d in product(range(2), range(12, 22), range(1, 11)):
        visual_state_values[b][p - 12][d - 1] = state_values[(bool(b), p, d)]
    for b in range(2):
        plt.imshow(visual_state_values[b])
        plt.show()


def main(trials=10000, lmbda=1):
    states = {(b, p, d) for b, p, d in product(
        (False, True),
        range(12, 22),
        range(1, 11))}
    state_values = {s: 0 for s in states}
    state_trials = {s: 0 for s in states}
    # Could use a bunch of DefaultDicts too but fuck that

    for _ in range(trials):
        player_cards = [card(), card()]
        dealer_cards = [card(), card()]
        while cardsum(player_cards) < 12:
            player_cards.append(card())
        state = cards_to_state(player_cards, dealer_cards)
        reward = simulate_and_get_reward(player_cards, dealer_cards)
        state_values[state] = (
            state_values[state] * state_trials[state] + reward) / (state_trials[state] + 1)
        state_trials[state] += 1

    visualize_state_values(state_values)


if __name__ == '__main__':
    main()
