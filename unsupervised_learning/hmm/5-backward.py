#!/usr/bin/env python3
"""Performs the backward algorithm for a hidden Markov Model"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a Hidden Markov Model.

    Parameters:
    - Observation (numpy.ndarray of shape (T,)): Index of the observation.
    - Emission (numpy.ndarray of shape (N, M)): Emission probabilities.
    - Transition (numpy.ndarray of shape (N, N)): Transition probabilities.
    - Initial (numpy.ndarray of shape (N, 1)): Initial probabilities.

    Returns:
    - P (float): Likelihood of the observations given the model.
    - B (numpy.ndarray of shape (N, T)): Backward path probabilities.
    """
    try:
        if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
            return None, None
        if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
            return None, None
        if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
            return None, None
        if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
            return None, None

        T = Observation.shape[0]  # Number of observations
        N = Emission.shape[0]     # Number of hidden states

        # Initialize backward path probabilities matrix
        B = np.zeros((N, T))
        B[:, -1] = 1

        # Fill the backward matrix in reverse
        for t in range(T - 2, -1, -1):
            for i in range(N):
                B[i, t] = np.sum(
                    B[:, t + 1]
                    * Transition[i, :]
                    * Emission[:, Observation[t + 1]]
                )

        # Compute the likelihood of the observations given the model
        P = np.sum(B[:, 0] * Initial[:, 0] * Emission[:, Observation[0]])

        return P, B

    except Exception as e:
        print(f"Error: {e}")
        return None, None
