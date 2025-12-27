import numpy as np

EPSILON = 1e-6


def discretize(features, feat_ranges, buckets_per_feat, n, i):
    '''
    features: shape [f]
    feat_ranges: shape [f, 2]
    buckets_per_feat: number of buckets along each feature axis
    k: n_buckets for a single feat
    n: n_tilings
    i: which tiling [0, n-1] it is
    We use the equation:
    int( [ ((f-a) / (b-a)) * kn/(n(k-1)+1) + i/kn ] * buckets_per_feat )
    '''

    features, feat_ranges, buckets_per_feat = map(
        np.asarray,
        (features, feat_ranges, buckets_per_feat))
    features = \
        (
            (features - feat_ranges[:, 0]) / (feat_ranges[:, 1] - feat_ranges[:, 0]) *
            (n * (buckets_per_feat - 1) + 1) / (buckets_per_feat * n) +
            i / (buckets_per_feat * n)) * \
        buckets_per_feat - EPSILON
    return features.astype(int)


def tile_encode(samples, feat_ranges, buckets_per_feat, n):
    '''
    samples: shape [n_samples, f]
    feat_ranges: shape [m, f]
    buckets_per_feat: number of buckets along each feature axis
    n: n_tilings
    '''
    samples, feat_ranges, buckets_per_feat = map(
        np.asarray,
        (samples, feat_ranges, buckets_per_feat))
    res = np.zeros((len(samples), n, np.product(buckets_per_feat)))
    tilemul = np.arange(n).reshape(-1, 1)
    dis = np.array([
        discretize(
            sample,
            feat_ranges,
            buckets_per_feat,
            n,
            tilemul)
        for sample in samples])
    i, j = np.indices(dis.shape[:2])
    res[
        i.ravel(),
        j.ravel(),
        np.ravel_multi_index(
            dis.reshape(
                -1,
                len(buckets_per_feat)).T,
            buckets_per_feat)] = 1
    return res


def hash_1darray(arr):
    # NOTE: This function depends on python's random hash seed.
    return hash(arr.data.tobytes())


def hash_encode(
        samples,
        feat_ranges,
        buckets_per_feat,
        n,
        hash_table_size,
        hash_fn=hash_1darray):
    samples, feat_ranges, buckets_per_feat = map(
        np.asarray,
        (samples, feat_ranges, buckets_per_feat))
    tilemul = np.arange(n).reshape(-1, 1)
    dis = np.array([
        hash_fn(
            discretize(
                sample,
                feat_ranges,
                buckets_per_feat,
                n,
                tilemul))
        for sample in samples]) % hash_table_size
    return dis


def main():
    import matplotlib.pyplot as plt
    from itertools import product
    from collections import defaultdict
    n = 4

    print(hash_1darray(np.array([0, 1])))
    print(hash_encode([[0, 0.25], [0.25, 0.75]], [[0, 1], [0, 1]], [3, 4], n, 69))

    import time
    t = time.time()
    X = tile_encode(np.random.uniform(0, 1, 300000).reshape(-1, 3), [[0, 1], [0, 1], [0, 1]], [7, 3, 4], n)
    print(f'Took {time.time() - t:.3f} seconds')
    print(X.shape)
    for i in range(n):
        buckets = defaultdict(list)
        for x, y in product(np.linspace(0, 1, 48), np.linspace(0, 1, 48)):
            bucket = hash(tuple(discretize([x, y], [[0, 1], [0, 1]], [7, 3], n, i)))
            buckets[bucket].append((x, y))

        fig = plt.figure()
        timer = fig.canvas.new_timer(interval=1000)
        timer.add_callback(lambda: plt.close())
        for bucket in buckets:
            plt.scatter(*zip(*buckets[bucket]), label=bucket)

        timer.start()
        plt.show()


if __name__ == '__main__':
    main()
