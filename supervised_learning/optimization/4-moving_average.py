#!/usr/bin/env python3
"""
Defines function that calculates the
weighted moving average of a data set
with bias correction
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set with bias correction.

    Args:
        data (list): The list of data points to calculate
        the moving average of.
        beta (float): The weight used for the moving average, between 0 and 1.

    Returns:
        list: A list containing the moving averages of the data.
    """
    moving_averages = []
    v = 0  # Initialize EMA to 0

    for i in range(len(data)):
        # Update EMA with current data point
        v = beta * v + (1 - beta) * data[i]
        # Apply bias correction
        bias_corrected_v = v / (1 - beta ** (i + 1))
        # Append the bias-corrected value to the list
        moving_averages.append(bias_corrected_v)
    return moving_averages
