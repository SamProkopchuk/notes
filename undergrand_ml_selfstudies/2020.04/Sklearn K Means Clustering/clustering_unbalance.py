import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt


def main():
    x = np.genfromtxt("unbalance.txt", delimiter=" ")
    clusters_chscores = []
    for clusters in range(2, 14):
        kmeans = KMeans(n_clusters=clusters).fit(x)
        clusters_chscores.append(
            (clusters, calinski_harabasz_score(x, kmeans.labels_)))

    plt.xlabel("Clusters")
    plt.ylabel("Calinski-Harabasz Score")
    plt.plot(
        [item[0] for item in clusters_chscores],
        [item[1] for item in clusters_chscores])
    plt.show()

    optimal_clusters = max(clusters_chscores, key=lambda x: x[1])[0]
    print(f"Optimal clusters: {optimal_clusters}")
    kmeans = KMeans(n_clusters=optimal_clusters).fit(x)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x[:, 0], x[:, 1], 'bo')
    plt.plot(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1], 'ro')
    plt.show()

if __name__ == "__main__":
    main()
