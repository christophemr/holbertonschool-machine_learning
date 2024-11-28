#!/usr/bin/env python3
"""
Viterbi Algorithm for Hidden Markov Models
"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a hidden
    Markov model.

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
        path (list): Length T, the most likely sequence of hidden states.
        P (float): Probability of obtaining the path sequence.
    """
    try:
        # Validate input dimensions
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

        # Initialize the Viterbi table and backpointer
        V = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)

        # Initialize the first column of V
        V[:, 0] = Initial.T * Emission[:, Observation[0]]

        # Populate the Viterbi table
        for t in range(1, T):
            for j in range(N):
                transition_probs = V[:, t - 1] * Transition[:, j]
                V[j, t] = (
                    np.max(transition_probs) * Emission[j, Observation[t]])
                backpointer[j, t] = np.argmax(transition_probs)

        # Backtrack to find the most likely path
        path = [np.argmax(V[:, T - 1])]
        for t in range(T - 1, 0, -1):
            path.insert(0, backpointer[path[0], t])

        # Compute the probability of the most likely path
        P = np.max(V[:, T - 1])

        return path, P
    except Exception as e:
        print(f"Error during computation: {e}")
        return None, None
