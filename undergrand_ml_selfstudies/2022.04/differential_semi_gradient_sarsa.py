'''
The cart pole problem, continuous.
Solved via sarsa using neural network state-action value approximation.
'''
import random
import gym
import numpy as np
import torch

from tqdm import tqdm

# Params you can't f with:

# ZERO THROTTLE: 0
# FULL THROTTLE FORWARD: 1
_NUM_ACTIONS = 2
_STATE_SHAPE = (4, )

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(_STATE_SHAPE[0] << 1, 256),
            torch.nn.Linear(256, 1)])
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            torch.nn.init.kaiming_uniform_(layer.weight)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.leaky_relu(x)
        return x

    def __call__(self, s, a):
        sh = _STATE_SHAPE[0]
        x = np.zeros(sh<<1)
        x[a * sh: (a+1) * sh] = s
        x = torch.as_tensor(x, dtype=torch.float32)
        return super().__call__(x)


def best_action(s, nn):
    return max(range(2), key=lambda a: nn(s, a))


def fit(trials=50000, epsilon=0.1, alpha=1e-6, beta=0.05, lmbda=0.95, lr=0.0001):
    # R is estimate of the average reward
    R = 0
    env = gym.make('CartPole-v1')
    env._max_episode_steps = np.inf
    nn = NN().float()
    optimizer = torch.optim.Adam(nn.parameters(), lr=lr)
    loss = torch.nn.MSELoss()
    ep_lens = [0]
    s, a = env.reset(), random.randint(0, 1)
    for i in tqdm(range(trials)):
        sprime, r, is_episode_done, _ = env.step(a)
        aprime = best_action(s, nn)
        if random.random() < epsilon:
            aprime = 1 - aprime
        r = -1 if is_episode_done else 0
        with torch.no_grad():
            pred = r + nn(sprime, aprime).item()
        y = nn(s, a)
        d = (r - R + pred - y).item()
        # l = loss(pred, R + nn(s, a))
        # optimizer.zero_grad()
        # l.backward()
        # optimizer.step()
        y.backward()
        with torch.no_grad():
            for param in nn.parameters():
                b4 = param.data.detach()
                param.data = param.data + alpha * d * param.grad.detach()
                if torch.isnan(param).all():
                    print(b4)
                    print(alpha * d * param.grad)
                    print(' <| |> ')
                    print(param)
                    exit(0)
        R = R + beta * d
        if is_episode_done:
            s = env.reset()
            a = best_action(s, nn)
            if random.random() < epsilon:
                a = 1 - a
            if len(ep_lens) > 10:
                print(f'Avg episode length: {sum(ep_lens) / len(ep_lens)}')
                ep_lens = [0]
                with torch.no_grad():
                    print(nn(s, 0), nn(s, 1))
                    for param in nn.parameters():
                        print(param)
                i = 0
            else:
                ep_lens.append(0)
        else:
            s, a = sprime, aprime
            ep_lens[-1] += 1
    return nn


def render_episodes(w, nepisodes=3):
    env = gym.make('CartPole-v1')
    env._max_episode_steps = np.inf
    for _ in range(nepisodes):
        s = env.reset()
        done = False
        while not done:
            a = best_action(s, w)
            s, r, done, d = env.step(a)
            env.render()
    env.close()


def main():
    w = fit()
    render_episodes(w)


if __name__ == '__main__':
    main()
