import numpy as np
from General_NN_Functions import sigmoid, tanh, relu, derivative, cost, evaluate_model

"""
A numpy implementation of all necessary functions for
a single-hidden-layer neural network.
"""


def layer_sizes(X, Y):
    input_size = X.shape[0]
    output_size = Y.shape[0]
    hidden_size = (input_size + output_size) // 2

    return (input_size, hidden_size, output_size)


def initialize_params(input_size, hidden_size, output_size,
                      multiplier=0.01, random_seed=2):
    np.random.seed(random_seed)
    W1 = np.random.randn(hidden_size, input_size) * multiplier
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * multiplier
    b2 = np.zeros((output_size, 1))

    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return params


def forward_propagate(X, params, funcs):
    # Get params from params dict
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    L1_func = funcs["L1_func"]
    L2_func = funcs["L2_func"]

    # Calculate Z1, A1, Z2, A2:
    Z1 = np.dot(W1, X) + b1
    A1 = L1_func(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = L2_func(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return cache


def backward_propagate(params, cache, funcs, X, Y):
    """
    Returns gradients (mean loss) of params relative to cost
    """
    m = X.shape[1]

    # Get W1, W2 from params:
    W1 = params["W1"]
    W2 = params["W2"]

    # Get A1, A2, Z1, Z2 from cache:
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]

    # Get layer 1 & 2 functions:
    L1_func = funcs["L1_func"]
    L2_func = funcs["L2_func"]

    # Calculate gradients:
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * derivative(f=L1_func, fx=L1_func(Z1))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return grads


def update_params(params, grads, learning_rate):
    # Get W1, b1, W2, b2 from params:
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    # Get dW1, db1, dW2, db2 from grads:
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return params


def fit_shl_nn_model(
        X, Y, funcs,
        learning_rate, num_iterations, cost_calc_interval,
        hidden_size=None, random_seed=3):
    costs = []

    if hidden_size is None:
        input_size, hidden_size, output_size = layer_sizes(X, Y)
    else:
        input_size, _, output_size = layer_sizes(X, Y)

    params = initialize_params(
        input_size, hidden_size, output_size, random_seed=random_seed)

    for i in range(num_iterations):
        cache = forward_propagate(X, params, funcs)
        grads = backward_propagate(params, cache, funcs, X, Y)
        params = update_params(params, grads, learning_rate)
        if i % cost_calc_interval == 0:
            costs.append(cost(cache["A2"], Y))
            print(f"{round(100*i/num_iterations)}% Complete", flush=True, end="\r")
    print("100% Complete")

    return params, costs


def predict(params, funcs, X):
    cache = forward_propagate(X, params, funcs)

    predictions = np.amax(cache["A2"], axis=0, keepdims=True)
    predictions = np.where(cache["A2"] == predictions, 1, 0)

    return predictions
