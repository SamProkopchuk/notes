import numpy as np
from scipy.stats import norm
from functools import reduce
from operator import mul
"""
This is a primitive anomaly detection algorithm example.

It uses the product of probabilities of every feature
relative to a normal distribution of the dataset's features.

If this product is less than a pre-defined epsilon,
then the given example is assumed to be anomalous.
"""
ANOM_THRESHOLD = 3e-8


def main():
    data = np.genfromtxt("psytest.csv", delimiter=",", skip_header=1)
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    norms = [norm(means[i], stds[i]) for i in range(data.shape[1])]
    while input("Continue? [Y/N]:") in ("y", "Y", ""):
        tests = [int(input(f"Test {i+1} score:")) for i in range(data.shape[1])]
        example_norm = reduce(mul, (norms[i].pdf(
            tests[i]) for i in range(data.shape[1])))
        if example_norm < ANOM_THRESHOLD:
            print("This example is an anomaly")
        else:
            print("Each test score seems relatively normal")

if __name__ == "__main__":
    main()
