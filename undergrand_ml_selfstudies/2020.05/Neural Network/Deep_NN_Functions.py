import numpy as np
from General_NN_Functions import sigmoid, tanh, relu, derivative, cost, evaluate_model

"""
A numpy implementation of all necessary functions for
a neural network with an arbitrary number of arbitrarily sized hidden layers. 
"""


def default_layer_sizes(X, Y, hidden_layers):
    hidden_layer_size = (X.shape[0] + Y.shape[0]) // 2

    return [X.shape[0]] + [hidden_layer_size] * hidden_layers + [Y.shape[0]]

def initialize_layer_sizes(
        X, Y, hidden_layers, hidden_layer_sizes):
    if hidden_layer_sizes is None:
        return default_layer_sizes(X, Y, hidden_layers)
    else:
        return [X.shape[0]] + list(hidden_layer_sizes) + [Y.shape[0]]

def initialize_params(layer_sizes, random_seed=None, multiplier=0.01):
    if random_seed is not None:
        np.random.seed(random_seed)

    params = {}

    for l in range(1, len(layer_sizes)):
        params[f"W{l}"] = np.random.randn(layer_sizes[l], layer_sizes[l - 1]) * multiplier
        params[f"b{l}"] = np.zeros((layer_sizes[l], 1))

    return params


def forward_propagate(X, params, funcs):
    # "A0" will point to X (no copy is made)
    cache = {"A0": X}

    for l in range(1, 1 + len(params) // 2):
        # Calculate Z{l} using A{l-1} (& W{l})
        cache[f"Z{l}"] = np.dot(params[f"W{l}"], cache[f"A{l-1}"]) + params[f"b{l}"]
        # Calculate A{l} using the activation function of layer l on Z{l}
        cache[f"A{l}"] = funcs[f"L{l}_func"](cache[f"Z{l}"])

    return cache


def backward_propagate(params, cache, funcs, X, Y):
    m = X.shape[1]
    L = len(params) // 2

    grads = {}

    # Do first back prob separately, since calculation for dZL is different:
    dZl = cache[f"A{L}"] - Y
    grads[f"dW{L}"] = np.dot(dZl, cache[f"A{L-1}"].T) / m
    grads[f"db{L}"] = np.sum(dZl, axis=1, keepdims=True) / m

    for l in range(L - 1, 0, -1):
        # Define lth function for cleaner code:
        funcl = funcs[f"L{l}_func"]

        # perform a backprop step:
        dZl = np.dot(params[f"W{l+1}"].T, dZl) * derivative(
            f=funcl, fx=funcl(cache[f"Z{l}"]))
        grads[f"dW{l}"] = np.dot(dZl, cache[f"A{l-1}"].T) / m
        grads[f"db{l}"] = np.sum(dZl, axis=1, keepdims=True) / m

    return grads


def update_params(params, grads, learning_rate):
    L = len(params) // 2

    for l in range(L, 0, -1):
        params[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        params[f"b{l}"] -= learning_rate * grads[f"db{l}"]

    return params


def fit_deep_nn_model(
        X, Y, hidden_layers, funcs, learning_rate, num_iterations,
        cost_calc_interval, hidden_layer_sizes=None, random_seed=None):
    """
    Notable args:
    funcs: dict of functions for every layer
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    costs = []

    layer_sizes = initialize_layer_sizes(
        X, Y, hidden_layers, hidden_layer_sizes)

    params = initialize_params(layer_sizes, random_seed=random_seed)
    L = len(params) // 2

    for i in range(num_iterations):
        cache = forward_propagate(X, params, funcs)

        grads = backward_propagate(params, cache, funcs, X, Y)

        params = update_params(params, grads, learning_rate)

        if i % cost_calc_interval == 0:
            costs.append(cost(cache[f"A{L}"], Y))
            print(f"{round(100*i/num_iterations)}% Complete", flush=True, end="\r")
    print("100% Complete")

    return params, costs


def predict(params, funcs, X):
    L = len(params) // 2
    cache = forward_propagate(X, params, funcs)

    predictions = np.amax(cache[f"A{L}"], axis=0, keepdims=True)
    predictions = np.where(cache[f"A{L}"] == predictions, 1, 0)

    return predictions
