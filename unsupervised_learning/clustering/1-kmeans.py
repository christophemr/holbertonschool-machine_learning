#!/usr/bin/env python3
"""
Function that performs K-means Clustering on a dataset
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Parameters:
        X (numpy.ndarray): Dataset of shape (n, d)
        k (int): Number of clusters
        iterations (int): Maximum number of iterations

    Returns:
        C (numpy.ndarray): Centroids of shape (k, d)
        clss (numpy.ndarray): Cluster assignments of shape (n,)
    """
    # Validate inputs
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Initialize centroids randomly within the data range
    n, d = X.shape
    low, high = np.min(X, axis=0), np.max(X, axis=0)
    centroids = np.random.uniform(low, high, size=(k, d))

    for _ in range(iterations):
        # Assign each data point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clss = np.argmin(distances, axis=1)

        # Recalculate centroids as the mean of assigned points
        new_centroids = np.empty_like(centroids)
        for i in range(k):
            points_in_cluster = X[clss == i]
            if points_in_cluster.size == 0:
                # Reinitialize empty cluster centroid
                new_centroids[i] = np.random.uniform(low, high, size=d)
            else:
                new_centroids[i] = points_in_cluster.mean(axis=0)

        # Stop if centroids do not change
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break

        # Update centroids for next iteration
        centroids = new_centroids

    return centroids, clss
