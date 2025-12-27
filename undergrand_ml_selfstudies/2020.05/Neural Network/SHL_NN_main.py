from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from General_NN_Functions import relu, tanh, sigmoid, evaluate_model, graph_costs
from SHL_NN_Functions import fit_shl_nn_model, predict
"""
Run an example of a single-hidden-layer neural network,
using necessary functions from SHL_NN_Functions.py
"""

# Parameters & Micro-parameters:
LEARNING_RATE = 0.3
NUM_ITERATIONS = 1000
COST_CALC_INTERVAL = 10
FUNCS = {"L1_func": relu, "L2_func": sigmoid}


def main():
    data = load_digits()
    X_orig = data.data
    Y_orig = data.target.reshape(-1, 1)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_orig, Y_orig, test_size=0.2, random_state=12)

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
    params, costs = fit_shl_nn_model(
        X_train, Y_train, FUNCS,
        learning_rate=LEARNING_RATE,
        num_iterations=NUM_ITERATIONS,
        cost_calc_interval=COST_CALC_INTERVAL)
    graph_costs(
        costs,
        x_label=f"{COST_CALC_INTERVAL}s of iterations",
        y_label="Mean loss (Cost)")
    predictions = predict(params, FUNCS, X_test)
    accuracy = evaluate_model(predictions, Y_test)

    print(f"The model acheived {accuracy*100:.2f}% accuracy on the test set.")


if __name__ == "__main__":
    main()
