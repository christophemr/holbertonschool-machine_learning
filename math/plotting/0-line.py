#!/usr/bin/env python3
"""plots y as a line graph"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """plots y as line graph"""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(y, color='red')
    plt.xlim(0, 10)
    plt.show()


if __name__ == "__main__":
    line()
