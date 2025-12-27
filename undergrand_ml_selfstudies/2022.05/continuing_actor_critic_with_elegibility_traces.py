"""
The cart pole problem, continuous.
Solved via Actor-Critic Algorithm with Elegibility Traces (continuing)
The policy pi uses a neural-network function H. Thereafter applying softmax.
The state-value v function is simple linear regression using tiling.
"""
import random
import gym
import numpy as np
import torch

from tqdm import tqdm
from tile import tile_encode

_NUM_ACTIONS = 2
_STATE_SHAPE = (4,)


_BUCKETS_PER_FEAT = [4, 4, 8, 8]

_FEAT_RANGES = [[-4.8, 4.8], [-3, 3], [-0.418, 0.418], [-3, 3]]
_N_TILINGS = 8


def clip_state(s):
    for i, fr in enumerate(_FEAT_RANGES):
        s[i] = min(fr[1], max(fr[0], s[i]))
    return s


class H(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(_STATE_SHAPE[0] * _NUM_ACTIONS, 32),
                torch.nn.Linear(32, 1),
            ]
        )
        self.activation_functions = torch.nn.ModuleList(
            [torch.nn.LeakyReLU(), torch.nn.LeakyReLU()]
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            torch.nn.init.kaiming_uniform_(layer.weight)

    def forward(self, x):
        for layer, activation_function in zip(self.layers, self.activation_functions):
            x = activation_function(layer(x))
        return x

    def __call__(self, s, a):
        x = np.zeros(_STATE_SHAPE[0] * _NUM_ACTIONS)
        x[a * _STATE_SHAPE[0] : (a + 1) * _STATE_SHAPE[0]] = s
        x = torch.as_tensor(x, dtype=torch.float32)
        return super().__call__(x)


class Policy:
    def __init__(self, h_func, lmbda, alpha):
        self.h = h_func
        self.lmbda = lmbda
        self.alpha = alpha
        self.optim = torch.optim.Adam(self.h.parameters(), maximize=True)
        self.z = 0

    def __call__(self, s):
        return self.best_action(s)

    def get_action_with_p(self, s):
        """
        Return a tuple of two items:
            a: The action according to the policy
            p: The pytorch-differentiable probability of a
        """
        a2p = [self.h(s, a) for a in range(_NUM_ACTIONS)]
        ps = [p.item() for p in a2p]
        tot = sum(np.exp(p) for p in ps)
        weights = [np.exp(p) / tot for p in ps]
        try:
            a = random.choices(range(_NUM_ACTIONS), weights=weights)[0]
        except ValueError as e:
            from os import get_terminal_size

            c = get_terminal_size().columns
            print(e)
            print("Debugging info:")
            print(a2p)
            print(ps)
            print(weights)
            print(tot)
            print("-" * c)
            for param in self.h.parameters():
                print(param.data)
            print("-" * c)
            exit(0)
        return a, torch.exp(a2p[a]) / tot

    def best_action(self, s):
        """
        Return the action corresponding to the highest probability
        """
        return max(range(2), key=lambda a: self.h(s, a))

    def update_params(self, d, gradient):
        self.update_elegibility_trace(gradient)
        negloss = d * gradient
        self.optim.zero_grad()
        negloss.backward()
        self.optim.step()

    def update_elegibility_trace(self, gradient):
        self.z = self.lmbda * self.z + gradient


class StateValueFunction:
    def __init__(self, buckets_per_feat, n_tiles, lmbda, alpha):
        self.buckets_per_feat = buckets_per_feat
        self.n_tiles = n_tiles
        self.w = np.zeros(self.n_tiles * np.product(self.buckets_per_feat))
        self.lmbda = lmbda
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.z = 0
        self.grad = None

    def __call__(self, s):
        s = clip_state(s)
        te = tile_encode([s], _FEAT_RANGES, self.buckets_per_feat, self.n_tiles)
        te = te.ravel()
        self.grad = te
        return self.w.dot(te)

    def update_params(self, d):
        self.update_elegibility_trace()
        self.w += self.alpha * d * self.z

    def update_elegibility_trace(self):
        self.z = self.lmbda * self.z + self.grad


def fit(pi, v, alpha=5e-3, trials=100_000):
    """
    Run the algorithm with:
        Policy pi
        State-value function v
    alpha: update rate for R
    """
    # R is estimate of the average reward
    R = 0
    env = gym.make("CartPole-v1")
    env._max_episode_steps = np.inf
    s, _ = env.reset()
    ep_lens = [0]
    for i in (pbar := tqdm(range(trials))):
        a, y = pi.get_action_with_p(s)
        sprime, r, is_episode_done, _, _ = env.step(a)
        d = r - R + v(sprime) - v(s)
        R += alpha * d
        v.update_params(d)
        pi.update_params(d, y)
        if is_episode_done:
            pbar.set_description(
                f"Avg Episode Length: {sum(ep_lens[-10:]) / len(ep_lens[-10:])} {R}"
            )
            ep_lens.append(0)
            s, _ = env.reset()
        else:
            ep_lens[-1] += 1
            s = sprime


def render_episodes(pi, nepisodes=3):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env._max_episode_steps = np.inf
    for _ in range(nepisodes):
        s, _ = env.reset()
        is_episode_done = False
        while not is_episode_done:
            a = pi.best_action(s)
            s, r, is_episode_done, d, _ = env.step(a)
            env.render()
    env.close()


def main():
    pi = Policy(h_func=H().float(), lmbda=0.80, alpha=1e-1)
    v = StateValueFunction(_BUCKETS_PER_FEAT, _N_TILINGS, lmbda=0.80, alpha=1e-2)
    fit(pi, v)
    render_episodes(pi)


if __name__ == "__main__":
    main()
