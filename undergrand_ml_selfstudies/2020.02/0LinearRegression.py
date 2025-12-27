import numpy as np
import matplotlib.pyplot as plt
import random as rand


def h(thet0, thet1, x):
    return thet0 + thet1 * x


def main():
    x = np.linspace(0, 10, 21)
    y = np.vectorize(lambda x: x + 5 + rand.uniform(-5, 5))(x)

    alpha = 0.05
    m = len(x)
    thet0 = 0
    thet1 = 0
    converged = False

    while not converged:
        regsum_thet0 = sum(h(thet0, thet1, x[i]) - y[i] for i in range(m))
        regsum_thet1 = sum(x[i] * (h(thet0, thet1, x[i]) - y[i])
                           for i in range(m))
        thet0 = thet0 - ((alpha / m) * regsum_thet0)
        thet1 = thet1 - ((alpha / m) * regsum_thet1)
        converged = (abs(regsum_thet0) + abs(regsum_thet1)) < 0.0001

    print("h(x) = {} + {}*x".format(thet0, thet1))

    plt.plot(x, y, 'bo')
    plt.plot(x, np.vectorize(lambda x: h(thet0, thet1, x))(x))
    plt.axis([-1, 11, 0, 20])
    plt.xlabel("x - axis")
    plt.ylabel("y - axis")
    plt.show()


if __name__ == "__main__":
    main()
    print('finished!')
