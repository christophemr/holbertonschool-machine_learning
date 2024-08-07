#!/usr/bin/env python3
""" plots all 5 previous graphs in one figure """
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    Plots all 5 previous graphs in one figure
    All axis labels and plot titles have a font size of x-small
    The plots are arranged in a 3 x 2 grid
    The last plot takes up two column widths
    The title of the figure is 'All in One'
    """
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    fig = plt.figure()

    # Graphique linéaire
    ax_line = fig.add_subplot(3, 2, 1)
    ax_line.plot(y0, 'r-')
    ax_line.set_xlim((0, 10))
    ax_line.set_ylim((0, 1000))
    ax_line.set_yticks([0, 500, 1000])

    # Graphique de dispersion
    ax_scatter = fig.add_subplot(3, 2, 2)
    ax_scatter.scatter(x1, y1, c='m')
    ax_scatter.set_xlabel('Height (in)', fontsize='x-small')
    ax_scatter.set_ylabel('Weight (lbs)', fontsize='x-small')
    ax_scatter.set_title("Men's Height vs Weight", fontsize='x-small')

    # Graphique de déclin exponentiel
    ax_decay = fig.add_subplot(3, 2, 3)
    ax_decay.plot(x2, y2)
    ax_decay.set_xlabel('Time (years)', fontsize='x-small')
    ax_decay.set_ylabel('Fraction Remaining', fontsize='x-small')
    ax_decay.set_title("Exponential Decay of C-14", fontsize='x-small')
    ax_decay.set_yscale("log")
    ax_decay.set_xlim((0, 28650))

    # Graphique avec deux courbes
    ax_two_lines = fig.add_subplot(3, 2, 4)
    ax_two_lines.plot(x3, y31, 'r--', label='C-14')
    ax_two_lines.plot(x3, y32, 'g-', label='Ra-226')
    ax_two_lines.set_xlabel('Time (years)', fontsize='x-small')
    ax_two_lines.set_ylabel('Fraction Remaining', fontsize='x-small')
    ax_two_lines.set_title("Exponential Decay of Radioactive Elements",
                           fontsize='x-small')
    ax_two_lines.legend(fontsize='x-small')
    ax_two_lines.set_xlim((0, 20000))
    ax_two_lines.set_ylim((0, 1))

    # Histogramme
    ax_hist = fig.add_subplot(3, 1, 3)
    ax_hist.hist(student_grades,
                 bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                 edgecolor='black')
    ax_hist.set_xlabel('Grades', fontsize='x-small')
    ax_hist.set_ylabel('Number of Students', fontsize='x-small')
    ax_hist.set_title("Project A", fontsize='x-small')
    ax_hist.set_xlim((0, 100))
    ax_hist.set_xticks(np.arange(0, 101, 10))
    ax_hist.set_ylim((0, 30))
    ax_hist.set_yticks(range(0, 30, 10))

    fig.suptitle("All in One", fontsize='x-small')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    all_in_one()
