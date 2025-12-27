import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

"""
Linear regression doesn't work well on this dataset
That said, the point of this is to use the idea of
stochastic gradient decent.
"""

DATA_PATH = "adult/adult.data"
COLUMN_LABELS = pd.Series(data=(
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "class"))
Y_LABEL = COLUMN_LABELS[0]
LR = 3e-5
ITERS_COST = 1000  # SGD iterations per mean-cost eval
CONV_THRESHOLD = 1e-2


def get_data():
    try:
        adult_df = pd.read_table(
            DATA_PATH, sep=", ", header=None, names=COLUMN_LABELS, engine="python")
    except FileNotFoundError:
        print("The required datasets were not found!")
        exit()
    return adult_df


def h(
        t: np.array,
        x: np.array):
    return x.dot(t)


def cost(
        t: np.array,
        x: np.array,
        y: np.array):
    return (h(t, x) - y)**2 / 2


def main():
    df = get_data()

    # Delete all rows that have "?".
    # There's almost definitely a better way of doing this.
    for col, dtype in enumerate(df.dtypes):
        if dtype == np.object_:
            df = df[df.iloc[:, col] != "?"]
    # But it works, and quickly enough.

    enc = OneHotEncoder()
    data = enc.fit_transform(df.select_dtypes(
        include=np.object_).to_numpy()).toarray()
    y_data = df[Y_LABEL].to_numpy()
    df.drop(Y_LABEL, axis=1, inplace=True)
    scaler = StandardScaler()
    x_data = np.c_[
        data,
        scaler.fit_transform(df.select_dtypes(exclude=np.object_).to_numpy())]
    x_data = np.c_[np.ones(x_data.shape[0]), x_data]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.20, random_state=12)
    # train_test_split already shuffles data, so no need to do so again
    # (As required by SGD)

    t = np.zeros(x_train.shape[1])
    index = 0
    costs = []
    plt.xlabel(f"{ITERS_COST}s of SGD iterations")
    plt.ylabel(f"Mean Cost in last {ITERS_COST} iterations")
    for _ in range(100):
        temp_costs = np.zeros(ITERS_COST)
        for i in range(ITERS_COST):
            temp_costs[i] = cost(t, x_train[index], y_train[index])
            t -= LR * (h(t, x_train[index]) - y_train[index]) * x_train[i]
            index = 0 if index == y_train.shape[0] - 1 else index + 1
        costs.append(temp_costs.mean())
        plt.plot(range(len(costs)), costs, "b")
        plt.pause(0.05)
    plt.show()
    for i in range(y_test.shape[0]):
        print(f"Test {i}: Predicted: {h(t, x_test[i])}, Actual: {y_test[i]}")
    print(
        ("Note: The dataset's features aren't very linear, ") +
        ("so linear regression doesn't work very well.\n") +
        ("The point of this was to apply SGD."))


if __name__ == "__main__":
    main()
