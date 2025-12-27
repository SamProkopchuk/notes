import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from General_NN_Functions import relu, tanh, sigmoid, evaluate_model, graph_costs
from Deep_NN_Functions import fit_deep_nn_model, predict

"""
The basic parameters and micro-parameters for a deep NN:
"""

LEARNING_RATE = 0.1
NUM_ITERATIONS = 10000
HIDDEN_LAYERS = 2
HIDDEN_LAYER_SIZES = None  # If None, will be set to output of default_layer_sizes

# Num of iterations between which cost is calculated:
COST_CALC_INTERVAL = 1

# The functions for every layer:
FUNCS = {f"L{i}_func": relu for i in range(1, HIDDEN_LAYERS + 1)}
FUNCS[f"L{HIDDEN_LAYERS+1}_func"] = sigmoid

RANDOM_SEED = 7
#


def main():
    """
    8x8 MNIST is kind of simple for a deep NN, maybe try a different dataset...
    """
    data = load_digits()
    X_orig = data.data
    Y_orig = data.target.reshape(-1, 1)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_orig, Y_orig, test_size=0.2, random_state=11)

    # Scale X:
    X_train = X_train / 16 - 0.5
    X_test = X_test / 16 - 0.5

    # Make Y Binarized:
    lb = LabelBinarizer()
    Y_train = lb.fit_transform(Y_train)
    Y_test = lb.transform(Y_test)

    # Transpose X & Y as required by the model
    X_train, X_test = X_train.T, X_test.T
    Y_train, Y_test = Y_train.T, Y_test.T
    params, costs = fit_deep_nn_model(
        X_train, Y_train,
        hidden_layers=HIDDEN_LAYERS,
        funcs=FUNCS,
        learning_rate=LEARNING_RATE,
        num_iterations=NUM_ITERATIONS,
        cost_calc_interval=COST_CALC_INTERVAL,
        random_seed=RANDOM_SEED)
    graph_costs(
        costs,
        x_label=f"{COST_CALC_INTERVAL}s of iterations",
        y_label="Mean loss (Cost)")
    predictions = predict(params, FUNCS, X_test)
    accuracy = evaluate_model(predictions, Y_test)

    print(f"The model acheived {accuracy*100:.2f}% accuracy on the test set.")

if __name__ == "__main__":
    main()
