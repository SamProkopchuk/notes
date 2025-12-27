import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# This is simplified Multiclass Classification.
# It uses one-vs all logistic regression.

# Also I tried to minimise my use of sklearn and
# pandas as by mainly using numpy I'm hoping to
# gain a better understanding of these concepts.


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    scaler = StandardScaler()
    data = pd.read_csv("Iris.csv")
    irises = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")
    iris_encoding = {irises[i]: i for i in range(3)}
    data["label"] = data["label"].replace(iris_encoding)

    X_train = data[data.columns[:4]].to_numpy()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_train = np.c_[X_train, np.ones(X_train.shape[0])]

    y_train = data[data.columns[4]].to_numpy()

    alpha = 0.1
    m = X_train.shape[1]
    THETAS = np.zeros((3, m))

    for iris in irises:
        print("Binary logistic regression now being performed with respect to:")
        print(iris.replace("-", " "))
        iter_ = iris_encoding[iris]

        y_train_bin = np.where(y_train == iter_, 1, 0)

        converged = False

        while not converged:
            ddX_vector = X_train.T.dot(
                sigmoid(X_train.dot(THETAS[iter_])) - y_train_bin)
            THETAS[iter_] = THETAS[iter_] - (alpha / m) * ddX_vector
            converged = np.vectorize(abs)(ddX_vector).sum() < 0.001

    print("done!")

    while (input("Continue [Y/n]?: ") in ("y", "Y", "")):
        x_test = []

        for question in ("petal_length,petal_width,sepal_length,sepal_width".replace("_", " ").split(",")):
            x_test.append(float(input("What is the " + question + " (cm)?: ")))

        x_test = np.array(x_test)
        x_test = scaler.transform([x_test])[0]
        x_test = np.r_[x_test, 1]

        for i in range(3):
            print(str(sigmoid(x_test.dot(THETAS[i])) * 100) +
                  "% certain that the flower is: " + irises[i])

if __name__ == "__main__":
    main()
