#!/usr/bin/env python3
"""
function that uses epsilon-greedy to determine the next action
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Chooses the next action using the epsilon-greedy strategy.

    Args:
        Q (numpy.ndarray): The Q-table.
        state (int): The current state.
        epsilon (float): The probability of exploring
        (choosing a random action).

    Returns:
        int: The index of the chosen action.
    """
    p = np.random.uniform(0, 1)
    if p < epsilon:
        # Exploring
        idx_action = np.random.randint(Q.shape[1])
    else:
        # Exploiting
        idx_action = np.argmax(Q[state])
    return idx_action
