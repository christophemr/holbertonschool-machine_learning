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
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Initialize centroids using uniform distribution
    n, d = X.shape
    low, high = np.min(X, axis=0), np.max(X, axis=0)
    centroids = np.random.uniform(low, high, size=(k, d))

    for _ in range(iterations):
        # Assign points to the nearest centroid (Euclidian distances)
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clss = np.argmin(distances, axis=1)

        # Calculate new centroids or Reinitializes empty clusters
        new_centroids = np.array(
            [X[clss == i].mean(axis=0) if np.any(clss == i) else
                np.random.uniform(low, high, size=d) for i in range(k)])

        # Stop if centroids do not change
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, clss
