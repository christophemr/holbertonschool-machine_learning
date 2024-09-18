#!/usr/bin/env python3
"""
function that calculates the precision
for each class in a confusion matrix
"""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels.

    Returns:
        numpy.ndarray: Precision for each class, shape (classes,)
    """
    # True Positives: Diagonal of the confusion matrix
    true_positives = np.diag(confusion)
    # False Positives: Sum of each column (predicted class),
    # excluding the diagonal
    false_positives = np.sum(confusion, axis=0) - true_positives
    # Precision for each class
    precision = true_positives / (true_positives + false_positives)
    return precision
