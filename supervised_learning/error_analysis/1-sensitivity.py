#!/usr/bin/env python3
"""
function that calculates the sensitivity
for each class in a confusion matrix
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels.

    Returns:
        numpy.ndarray: Sensitivity for each class, shape (classes,)
    """
    # True Positives: Diagonal of the confusion matrix
    true_positives = np.diag(confusion)
    # False Negatives: Sum of each row (actual class), excluding the diagonal
    false_negatives = np.sum(confusion, axis=1) - true_positives
    # Sensitivity for each class
    sensitivity = true_positives / (true_positives + false_negatives)
    return sensitivity
