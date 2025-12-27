import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    scaler = StandardScaler()
    digits = load_digits()

    base10_digits = tuple(i for i in range(10))

    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.30, random_state=32)

    X_train = scaler.fit_transform(X_train)
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]

    alpha = 0.3
    m = X_train.shape[1]
    THETAS = np.zeros((10, m))

    converged = False

    for base10_digit in (base10_digits):
        print("Binary logistic regression now being performed with respect to the number: {}".format(
            base10_digit))

        y_train_bin = np.where(y_train == base10_digit, 1, 0)
        converged = False
        i = 0

        while not converged:
            ddX_vector = X_train.T.dot(
                sigmoid(X_train.dot(THETAS[base10_digit])) - y_train_bin)
            THETAS[base10_digit] = THETAS[
                base10_digit] - (alpha / m) * ddX_vector
            if i % 100 == 0:
                converged = np.vectorize(abs)(ddX_vector).sum() < 10
            if i % 5000 == 0:
                print("At {} iterations the derivative is {}".format(
                    i, np.vectorize(abs)(ddX_vector).sum()))
                # print("Max theta value is:", THETAS[base10_digit].max())
            i += 1

    wrongs = 0

    for x_test_index in range(X_test.shape[0]):
        x_test = np.r_[1, scaler.transform([X_test[x_test_index]])[0]]
        guesses = []
        for base10_digit in base10_digits:
            guesses.append(sigmoid(x_test.dot(THETAS[base10_digit])))
        guess = guesses.index(max(guesses))
        print("The model guesses {} and it was {}.".format(
            guess, y_test[x_test_index]))
        if guess != y_test[x_test_index]:
            wrongs += 1
            plt.gray()
            plt.matshow(X_test[x_test_index].reshape(8, 8))
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()

    print("Accuracy: {}%".format(100 * (len(y_test) - wrongs) / len(y_test)))


if __name__ == "__main__":
    main()
