#!/usr/bin/env python3
"""function that updates the learning rate using inverse time
decay in numpy"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay with NumPy.
    Args:
        alpha (float): The original learning rate.
        decay_rate (float): The rate at which alpha will decay.
        global_step (int): The number of passes of gradient descent
        that have elapsed.
        decay_step (int): The number of passes of gradient descent that
        should occur before alpha is decayed further.
    Returns:
        float: The updated value for alpha.
    """
    # Calculate the decay factor using NumPy floor division
    decay_factor = np.floor(global_step / decay_step)
    # Update the learning rate using the inverse time decay formula
    updated_alpha = alpha / (1 + decay_rate * decay_factor)
    return updated_alpha
