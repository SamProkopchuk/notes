import numpy as np


def main():
    data = np.genfromtxt('basketball.csv', delimiter=',', dtype=np.float_)
    data = np.delete(data, 0, axis=0)
    y = data[:, data.shape[1] - 1]
    X = np.delete(data, data.shape[1] - 1, axis=1)

    # Add a column of ones to X, such that W m+1 will be constant.
    X = np.c_[X, np.ones(X.shape[0])]

    W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    # Please note the following loop only works if the given data set has 4
    # x-values.
    while input("continue? [y/n]: ") in ("y", "Y", ""):
        print("We will now predict a final exam score, given:")
        print("(Please enter numerical values)")
        x1 = float(input("Height in feet: "))
        x2 = float(input("Weight in pounds: "))
        x3 = float(input("Percent of successful field goals (Decimal 0<=x<=1): "))
        x4 = float(input("Percent of successful free throws (Decimal 0<=x<=1): "))
        print("The predicted average points scored per game is:")
        print(np.array([x1, x2, x3, x4, 1]).dot(W))

if __name__ == "__main__":
    main()
    print('finished!')
