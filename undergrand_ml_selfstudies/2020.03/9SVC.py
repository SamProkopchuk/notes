import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm


def main():
    data = load_iris()

    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=12)

    clf = svm.SVC(gamma="auto")
    clf.fit(x_train, y_train)

    print(str(round(clf.score(x_test, y_test) * 100, 2)) +
          "% accurate on test set.")

if __name__ == "__main__":
    main()
