#!/usr/bin/env python3
"""
This script plots the exponential decay of two radioactive elements,
C-14 and Ra-226,
as line graphs using Matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    Plots the exponential decay of C-14 and Ra-226 as line graphs.
    The x-axis represents time in years, ranging from 0 to 20,000.
    The y-axis represents the fraction remaining, ranging from 0 to 1.
    The decay of C-14 is represented by a dashed red line.
    The decay of Ra-226 is represented by a solid green line.
    A legend is placed in the upper right corner to label the two lines.
    """

    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(x, y1, 'r--', label='C-14')
    plt.plot(x, y2, 'g-', label='Ra-226')
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title("Exponential Decay of Radioactive Elements")
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    two()
