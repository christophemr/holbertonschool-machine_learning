#!/usr/bin/env python3
"""
This module contains a function that determines if you should stop
gradient descent early.
"""
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should be stopped early.

    Parameters:
        cost (float): The current validation cost of the neural network.
        opt_cost (float): The lowest recorded validation cost.
        threshold (float): The threshold for early stopping.
        patience (int): The patience count for early stopping.
        count (int): The current count of how long the threshold
        has not been met.
    Returns:
        (bool, int): A tuple with a boolean indicating if
        early stopping should occur,
                     and the updated count.
    """
    # Check if the current cost is significantly lower than the
    # optimal cost plus threshold
    if cost < opt_cost - threshold:
        # Reset the count if there is a significant improvement
        return False, 0
    else:
        # Increment the count if there is no significant improvement
        count += 1
        # Check if the patience limit has been reached
        if count >= patience:
            return True, count
        else:
            return False, count
