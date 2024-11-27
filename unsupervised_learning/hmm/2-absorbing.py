#!/usr/bin/env python3
"""
Determines if a Markov chain is absorbing.
"""

import numpy as np


def absorbing(P):
    """
    Determines if a Markov chain is absorbing.

    Parameters:
        P (numpy.ndarray): Square 2D array of shape (n, n)
            - transition_matrix[i, j] is the probability of transitioning
            from state i to state j
            - n is the number of states in the Markov chain

    Returns:
        bool: True if the Markov chain is absorbing, False otherwise.
    """
    # Validate input
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    num_states, num_columns = P.shape
    if num_states != num_columns:
        return False
    if not np.all((0 <= P) & (P <= 1)):
        return False
    if not np.allclose(P.sum(axis=1), 1):
        return False

    # Identify absorbing states (diagonal elements == 1)
    absorbing_states_indices = np.where(np.diag(P) == 1)[0]

    # If no absorbing states, the chain is not absorbing
    if len(absorbing_states_indices) == 0:
        return False

    # Determine transient states (non-absorbing)
    transient_states_indices = np.setdiff1d(
        range(num_states), absorbing_states_indices)
    if len(transient_states_indices) == 0:
        return True  # All states are absorbing

    # Construct the submatrix Q
    transient_to_transient_matrix = P[np.ix_(
        transient_states_indices, transient_states_indices)]

    # Check if every transient state can reach an absorbing state
    try:
        # Identity matrix for transient states
        identity_matrix = np.eye(len(transient_states_indices))
        # Fundamental matrix: (I - Q)^-1
        fundamental_matrix = np.linalg.inv(
            identity_matrix - transient_to_transient_matrix)
        # Submatrix R: transient-to-absorbing transitions
        transient_to_absorbing_matrix = P[np.ix_(
            transient_states_indices, absorbing_states_indices)]
        # Verify that at least one absorbing state is reachable
        # from every transient state
        if np.all(np.sum(np.dot(fundamental_matrix,
                                transient_to_absorbing_matrix), axis=1) > 0):
            return True
    except np.linalg.LinAlgError:
        return False

    return False
