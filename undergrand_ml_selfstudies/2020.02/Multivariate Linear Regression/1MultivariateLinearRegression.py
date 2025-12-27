import numpy as np

"""
The matrix-math of this is based on the following:
https://vxy10.github.io/2016/06/25/lin-reg-matrix/
"""


def scale(vector, column_mean, column_range):
    return (vector - column_mean) / column_range


def main():
    data = np.genfromtxt('psytest.csv', delimiter=',', dtype=np.float_)
    data = np.delete(data, 0, axis=0)
    y = data[:, 3]
    X = np.delete(data, data.shape[1] - 1, axis=1)

    # x_scale is a numpy array of the means and ranges of each column.
    X_scale = np.stack((X.mean(axis=0), np.amax(
        X, axis=0) - np.amin(X, axis=0)), axis=0)

    # Now we can scale x:
    X = scale(X, X_scale[0], X_scale[1])

    # Add a column of ones to X, such that W m+1 will be constant.
    X = np.c_[X, np.ones(X.shape[0])]

    W = np.zeros(X.shape[1])

    alpha = 0.03
    converged = False

    while not converged:
        # Please reference the link at the top of the file if you're confused
        # regarding what the hell is being done here.
        derivative = W.T.dot(X.T.dot(X)) - y.T.dot(X)
        W = W - (alpha / X.shape[1]) * (derivative)
        converged = np.vectorize(abs)(derivative).sum() < 0.000001

    # Please note the following loop only works if the given data set has 3 x-values.
    # This could be generalized in a for loop - which is on my to do list.
    while input("continue? [y/n]:") in ("y", "Y", ""):
        print("We will now predict a final exam score, given:")
        print("(Please enter integer values)")
        ex1 = scale(int(input("Exam 1 score:")), X_scale[0][0], X_scale[1][0])
        ex2 = scale(int(input("Exam 2 score:")), X_scale[0][1], X_scale[1][1])
        ex3 = scale(int(input("Exam 3 score:")), X_scale[0][2], X_scale[1][2])
        print("The predicted final exam score is:")
        print(np.array([ex1, ex2, ex3, 1]).dot(W))

if __name__ == "__main__":
    main()
    print('finished!')
