#!/usr/bin/env python3
"""function that plots a histogram of student scores
for a project using matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram of student scores for a project.
    The x-axis is labeled 'Grades' and has bins every 10 units.
    The y-axis is labeled 'Number of Students'.
    The title of the histogram is 'Project A'.
    The bars of the histogram are outlined in black.
    """

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title("Project A")
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.xticks(range(0, 101, 10))
    plt.show()


if __name__ == "__main__":
    frequency()
