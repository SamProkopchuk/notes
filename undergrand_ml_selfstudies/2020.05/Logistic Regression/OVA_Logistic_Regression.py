from Logistic_Regression import sigmoid, get_hypothesis, get_cost, optimize, get_accuracy, graph
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


"""
Reviewing one-vs-all Logistic Regression
"""


def get_ova_y(y, one):
    one_y = np.where(y == one, 1, 0)
    return one_y


def ova_optimize(X, y, learning_rate, num_iterations):
    ts_bs, costs = {}, {}
    unique = np.unique(y, return_counts=False)
    for one in unique:
        t = np.zeros(X.shape[1])
        b = 0
        ova_y = get_ova_y(y, one)
        tpart, bpart, costpart = optimize(
            t, b, X, ova_y, learning_rate, num_iterations)
        ts_bs[one] = (tpart, bpart)
        costs[one] = costpart
    return ts_bs, costs


def get_hypotheses(ts_bs, X):
    hypotheses = {}
    for category in ts_bs:
        hypotheses[category] = get_hypothesis(
            ts_bs[category][0], ts_bs[category][1], X)
    return hypotheses


def get_norm_hypotheses(ts_bs, X):
    """
    Return hypotheses on X
    such that the probabilities sum to 1.
    """
    hypotheses = get_hypotheses(ts_bs, X)
    hyp_items = hypotheses.items()
    hyp_keys_values = list(zip(*hyp_items))
    hyp_values = np.array(hyp_keys_values[-1])
    hyp_values /= hyp_values.sum(axis=0, keepdims=True)
    hyp_items = zip(hyp_keys_values[0], tuple(hyp_values))
    hypotheses = dict(hyp_items)
    return hypotheses


def get_predicted(hypotheses):
    """
    Given hypotheses dict returns list of
    categories of highest corresponding hypotheses
    """
    categorized = []
    for i in range(len(list(hypotheses.values())[0])):
        key, _ = max(hypotheses.items(), key=lambda x: x[1][i])
        categorized.append(key)
    return categorized


def main():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.20, random_state=31)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    ts_bs, costs = ova_optimize(X_train, y_train, 3, 1000)
    graph(costs)
    norm_hypotheses = get_norm_hypotheses(ts_bs, X_test)
    categorized = get_predicted(norm_hypotheses)
    print(f"Accuracy on test set: {100*get_accuracy(y_test, categorized):.2f}%")

if __name__ == "__main__":
    main()
