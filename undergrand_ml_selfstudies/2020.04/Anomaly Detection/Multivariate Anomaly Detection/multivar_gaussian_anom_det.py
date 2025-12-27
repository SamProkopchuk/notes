import numpy as np
from scipy.stats import multivariate_normal as mv_norm


"""
Take advantage of the preassumed linear trend of exam scores
to better detect anomalies - using the multivariate normal distribution.
See: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
Or more pragmatically: check out Andrew Ng's course on Coursera lol..
"""

ANOM_THRESHOLD = 3e-8


def main():
    data = np.genfromtxt("psytest.csv", delimiter=",", skip_header=1)
    means = np.mean(data, axis=0)
    sigma = sum(partial for partial in np.apply_along_axis(
        lambda x: np.outer(x, x), axis=1, arr=data - means)) / data.shape[0]
    static_mv_graph = mv_norm(mean=means, cov=sigma)
    while input("Continue? [Y/N]:") in ("y", "Y", ""):
        tests = [int(input(f"Test {i+1} score:")) for i in range(data.shape[1])]
        test_prob = static_mv_graph.pdf(np.asarray(tests))
        if test_prob < ANOM_THRESHOLD:
            print("This example is an anomaly")
        else:
            print("This example didn't have a greatly anomalous scores")

if __name__ == "__main__":
    main()
