#!/usr/bin/env python3
"""
function that plots a stacked bar chart representing the
number of fruits various people possess
"""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plots a stacked bar graph representing the
    number of fruits various people possess.
    The columns of the fruit matrix represent the number of fruits
    Farrah, Fred, and Felicia have, respectively.
    The rows of the fruit matrix represent the number of apples
    , bananas, oranges, and peaches, respectively.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    people = ['Farrah', 'Fred', 'Felicia']
    fruit_names = {
        'apples': 'red',
        'bananas': 'yellow',
        'oranges': '#ff8000',
        'peaches': '#ffe5b4'
    }

    # Initialize the bottom values to zero
    bottom = np.zeros(len(people))

    # Create the figure with a specific size
    plt.figure(figsize=(6.4, 4.8))

    # Create the stacked bar chart
    for i, (name, color) in enumerate(sorted(fruit_names.items())):
        plt.bar(
            np.arange(len(people)),
            fruit[i],
            width=0.5,
            bottom=bottom,
            color=color,
            label=name
        )
        bottom += fruit[i]

    # Configure the axis labels, ticks, and title
    plt.xticks(np.arange(len(people)), people)
    plt.yticks(np.arange(0, 81, 10))
    plt.ylabel('Quantity of Fruit')
    plt.title("Number of Fruit per Person")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    bars()
