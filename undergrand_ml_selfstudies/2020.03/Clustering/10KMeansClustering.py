import numpy as np
import matplotlib.pyplot as plt

EPSILON = 1e-12


def main():
    data = np.genfromtxt("Iris.txt", skip_header=1, delimiter=",")
    x = np.c_[data[:, 1], data[:, 2]]
    # plt.plot(x[:, 0], x[:, 1], 'bo')
    # plt.show()

    k = int(input("Clusters: "))

    # Create a random number generating object for the initial k points.
    rng = np.random.default_rng()

    centroids = rng.choice(a=x, size=k, replace=False)
    distances = np.zeros((k, x.shape[0]))

    done = False
    while not done:
        for i in range(k):
            distances[i] = np.apply_along_axis(
                np.linalg.norm, axis=1, arr=x - centroids[i])
        print(distances.shape)
        # Get array of which pivot every points is closer to.
        closest = np.apply_along_axis(
            lambda x: np.where(x == np.amin(x))[0], axis=0, arr=distances)[0]
        print(closest)

        new_centroids = np.zeros((k, x.shape[1]))
        closest_count = np.zeros(k)

        for i in range(x.shape[0]):
            new_centroids[closest[i], :] += x[i, :]
            closest_count[closest[i]] += 1
        for i in range(k):
            if closest_count[i] != 0:
                new_centroids[i, :] /= closest_count[i]
        if abs((new_centroids - centroids)).sum() < EPSILON:
            break
        centroids = new_centroids.copy()

    plt.plot(x[:, 0], x[:, 1], 'bo')
    plt.plot(centroids[:,0], centroids[:,1], 'ro')
    plt.show()

if __name__ == "__main__":
    main()
