import numpy as np
import matplotlib.pyplot as plt


"""
Reviewing Logistic Regression
(With numpy instead of sklearn.linear_model.LogisticRegression)
"""


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def get_hypothesis(t, b, X):
    return sigmoid(np.dot(X, t) + b)


def get_cost(t, b, X, y):
    h = get_hypothesis(t, b, X)
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / (X.shape[0])


def get_grads(t, b, X, y):
    h = get_hypothesis(t, b, X)
    dt = np.dot(X.T, (h - y)) / (X.shape[0])
    db = np.sum(h - y) / (X.shape[0])
    return dt, db


def optimize(t, b, X, y, learning_rate, num_iterations):
    cost_over_iterations = []
    for i in range(num_iterations):
        cost_over_iterations.append(get_cost(t, b, X, y))
        dt, db = get_grads(t, b, X, y)
        t = t - learning_rate * dt
        b = b - learning_rate * db
    return t, b, cost_over_iterations


def get_accuracy(y_test, y_predict):
    total = len(y_test)
    correct = 0
    for test, predict in zip(y_test, y_predict):
        if test == predict:
            correct += 1
    return correct / total


def graph(costs):
    """
    Graphs each one vs all cost time-series
    given dict of o.v.a. cost time-series.
    """
    for one in costs:
        ova_cost = costs[one]
        plt.plot(range(len(ova_cost)), ova_cost, label=one)
    plt.legend(loc="upper right")
    plt.show()
