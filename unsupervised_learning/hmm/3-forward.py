#!/usr/bin/env python3
"""
Forward Algorithm for Hidden Markov Models
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden Markov model.

    Parameters:
        Observation (numpy.ndarray): shape (T,)
            Contains the index of the observations.
        Emission (numpy.ndarray): shape (N, M)
            Contains the emission probability of a specific observation
            given a hidden state.
        Transition (numpy.ndarray): shape (N, N)
            Contains the transition probabilities.
        Initial (numpy.ndarray): shape (N, 1)
            Contains the probability of starting in a particular hidden state.

    Returns:
        P (float): Likelihood of the observations given the model.
        F (numpy.ndarray): shape (N, T)
            Contains the forward path probabilities.
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

        if Transition.shape != (N, N) or Initial.shape != (N, 1):
            return None, None

        # Initialize forward probabilities
        F = np.zeros((N, T))
        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        # Compute forward probabilities for each time step
        for t in range(1, T):
            for j in range(N):
                F[j, t] = np.sum((F[:, t - 1] * Transition[:, j])
                                 * Emission[j, Observation[t]])

        # Compute the likelihood of the observation sequence
        P = np.sum(F[:, -1])

        return P, F
    except Exception as e:
        print(f"Error during computation: {e}")
        return None, None
