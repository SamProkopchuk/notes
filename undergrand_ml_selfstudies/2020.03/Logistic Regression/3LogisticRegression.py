import numpy as np

# This regression program is specifically tailored
# for the given dataset in the same folder.

"""

g(x) = 1
	--------
	1+e^-X.T.dot(THETA)

"""


def scale(vector, column_mean, column_range):
    return (vector - column_mean) / column_range


def g(x):
    return 1 / (1 + np.exp(-x))


def main():
    data = np.genfromtxt("Kid.csv", delimiter=",")
    data = np.delete(np.delete(data, 0, axis=0), 0, axis=1)
    y = data[:, 0]
    X = np.delete(data, 0, axis=1)

    # scales for all x values
    X_scale = np.stack((X.mean(axis=0), np.amax(
        X, axis=0) - np.amin(X, axis=0)), axis=0)

    X = scale(X, X_scale[0], X_scale[1])

    X = np.c_[X, np.ones(X.shape[0])]

    alpha = 0.03
    m = X.shape[1]
    THETA = np.zeros(m)
    converged = False
    i = 1

    while not converged:
        derivative_vector = X.T.dot(g(X.dot(THETA)) - y)
        THETA = THETA - (alpha / m) * derivative_vector
        converged = np.vectorize(abs)(derivative_vector).sum() < 0.001
        i += 1
    print("Num of iters: {}".format(i))

    while input("Continue [Y/n]?") in ("Y", "y", ""):
        prediction_X = []
        for question in "Income,Is Female,Is Married,Has College,Is Professional,\
        		Is Retired,Unemployed,Residence Length,Dual Income,Minors,Own,\
        		House,White,English,Prev Child Mag,Prev Parent Mag".split(","):
            prediction_X.append(float(input(question + "?: ")))
        prediction_X = np.array(prediction_X)
        prediction_X = scale(prediction_X, X_scale[0], X_scale[1])
        prediction_X = np.r_[prediction_X, 1]
        print("The probability that this individual will purchase is: ")
        print(g(prediction_X.dot(THETA)))


if __name__ == "__main__":
    main()
