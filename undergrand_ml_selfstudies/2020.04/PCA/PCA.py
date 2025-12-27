import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

VARIANCE_THRESHOLD = 1 - 1e-2


def main():
    data = pd.read_table("covtype.data", header=None, delimiter=",")
    train, test = train_test_split(data, test_size=0.2)
    scaler = StandardScaler()
    train = scaler.fit_transform(train)

    train_np = np.asarray(train)
    sigma = (train_np.T.dot(train_np)) / train_np.shape[0]
    u, s, v = np.linalg.svd(sigma)

    k = 0
    k_sum = 0
    m_sum = sum(s)
    while k_sum / m_sum < VARIANCE_THRESHOLD:
        k_sum += s[k]
        k += 1

    print((f"Dimention can be reduced from m={train.shape[1]} to "
    f"k={k}, maintaining {VARIANCE_THRESHOLD*100}% variance."))

    # Since the train matrix has m columns, we need to take
    # its transpose such that every data vector is a column.
    train_kdim = u[:, :k].T.dot(train.T)

    print(train_kdim)

if __name__ == "__main__":
    main()
