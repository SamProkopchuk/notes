import numpy as np
from math import ceil
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from scipy.special import expit, logit

# expit = sigmoid
# logit = inverse of expit

# This NN is simplified and poorly optimized
# To justify, this was done to simply get a better understanding
# of neural networks and back propogation.


def expit_deriv(x):
    return expit(x) * (1 - expit(x))


def gen_weights_from_nodes(NODES):
    WEIGHTS = []
    for l in range(1, NODES.shape[0]):
        weights = []
        j_range = NODES[l].shape[0] if l == NODES.shape[
            0] - 1 else NODES[l].shape[0] - 1
        for j in range(j_range):
            weights.append(
                np.random.rand(NODES[l - 1].shape[0]))
        WEIGHTS.append(np.asarray(weights))
    return np.asarray(WEIGHTS)


def forward_propagate(x, NODES, WEIGHTS):
    """
    Assume x, NODES, & WEIGHTS of proper dimentions are given.
    """
    NODES[0][1:] = x
    for l in range(1, NODES.shape[0]):
        has_bias = 1 if l != NODES.shape[0] - 1 else 0
        NODES[l][has_bias:] = expit(WEIGHTS[l - 1].dot(NODES[l - 1]))
    return NODES[NODES.shape[0] - 1]


def get_cost(X, Y, NODES, WEIGHTS):
    cost = 0
    for sample_no in range(X.shape[0]):
        cost += np.abs((forward_propagate(X[sample_no],
                                          NODES, WEIGHTS) - Y[sample_no]).sum())
    return cost


def back_propagate(DELTAS, yi_train, y_pred, NODES, WEIGHTS):
    deltas = []
    for array in NODES:
        deltas.append(np.zeros_like(array))
    deltas = np.asarray(deltas)

    deltas[-1] = y_pred - yi_train

    weights_count = WEIGHTS.shape[0]

    deltas[1] = WEIGHTS[1].T.dot(
        deltas[-1]) * expit_deriv(logit(NODES[1]))

    DELTAS[0] += np.outer(deltas[1], NODES[0])[1:]

    DELTAS[1] += np.outer(deltas[2], NODES[1])


def main():
    np.random.seed(32)
    scaler = StandardScaler()
    lb = LabelBinarizer()
    data = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.20, random_state=16)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    # Create a simple 3 layer NN:
    NODES = np.array((
        np.r_[1, np.zeros(X_train.shape[1])],
        np.r_[1, np.zeros(ceil(np.mean((X_train.shape[1], 3))))],
        np.zeros(3)))
    # Note the adding of an extra node in all but the last layer.

    WEIGHTS = gen_weights_from_nodes(NODES)
    DELTAS = []
    for weights in WEIGHTS:
        DELTAS.append(np.zeros_like(weights))
    DELTAS = np.asarray(DELTAS)

    iteration = 0
    alpha = 0.1
    while iteration <= 300:
        cost = get_cost(X_train, y_train, NODES, WEIGHTS)

        for i in range(DELTAS.shape[0] - 1):
            DELTAS[i] = np.zeros_like(DELTAS[i])

        for i in range(X_train.shape[0]):
            yi_pred = forward_propagate(X_train[i], NODES, WEIGHTS)
            back_propagate(DELTAS, y_train[i], yi_pred, NODES, WEIGHTS)

        WEIGHTS -= DELTAS * alpha / X_train.shape[0]

        if iteration % 30 == 0:
            print("\nNN state:")
            print("Cost relative to training set:", cost)
            correct = 0
            for i in range(y_test.shape[0]):
                guess = forward_propagate(X_test[i], NODES, WEIGHTS)
                if np.array_equal(
                        y_test[i], np.vectorize(lambda x: int(x == max(guess)))(guess)):
                    correct += 1
                # else:
                #     print(np.vectorize(lambda x: int(x == max(guess)))(guess))
                #     print(y_test[i])
            print("Accuracy relative to training set: about",
                  str(round(correct / y_test.shape[0] * 100)) + "%")
            print("\n")

        iteration += 1

if __name__ == "__main__":
    main()

# Time to move on to a nice NN library lol.
