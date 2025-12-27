import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def main():
    data = pd.read_csv("auto-mpg.csv", delimiter=",")
    X_train = data[data.columns[1:6]].to_numpy()
    # Concatenate Square of Features to X_train matrix:
    X_train = np.apply_along_axis(
        lambda x: np.concatenate([x, np.square(x)]), 1, X_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = np.c_[X_train, np.ones(X_train.shape[0])]
    print(X_train)
    Y_train = data[data.columns[0]].to_numpy()

    L = np.eye(X_train.shape[1])
    L[0, 0] = 0
    lamb = 5

    THETA = np.linalg.inv(X_train.T.dot(X_train) +
                          lamb * L).dot(X_train.T.dot(Y_train))
    print(THETA)

    X_test = []
    for question in "cylinders,displacement,horsepower,weight,acceleration".split(","):
        X_test.append(float(input(question + "?: ")))
    X_test = np.array(X_test)
    X_test = scaler.transform([X_test])[0]
    X_test = np.concatenate([X_test, np.square(X_test), np.ones(1)])
    print(X_test.dot(THETA))

if __name__ == "__main__":
    main()
