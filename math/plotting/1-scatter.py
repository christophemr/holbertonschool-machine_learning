#!/usr/bin/env python3
"""function that plots x & y as a scatter plot"""


import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """plots data as magenta points. y & w axis are labeled
    and the graph is titled"""
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    plt.scatter(x, y, color='magenta')
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.title("Men's Height vs Weight")
    plt.show()


if __name__ == "__main__":
    scatter()
