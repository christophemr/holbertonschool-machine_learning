#!/usr/bin/env python3
"""function nthat plots x, y as a line graph"""


import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """plots a line graph
    x-axis is labeled as 'time(years)'
    y-axis as 'fraction remaining'
    the title is 'exponential decay of c-14
    the y-axis is logarithmically scaled
    the x-axis ranges from 0 to 28650"""
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(x, y)
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title("Exponential Decay of C-14")
    plt.yscale("log")
    plt.xlim(0, 28650)
    plt.show()
