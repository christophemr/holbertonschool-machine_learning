#!/usr/bin/env python3
"""
function that calculates the specificity
for each class in a confusion matrix
"""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels.

    Returns:
        numpy.ndarray: Specificity for each class, shape (classes,)
    """
    # True Positives: Diagonal of the confusion matrix
    true_positives = np.diag(confusion)
    # False Positives: Sum of each column (predicted class),
    # excluding the diagonal
    false_positives = np.sum(confusion, axis=0) - true_positives
    # False Negatives: Sum of each row (actual class),
    # excluding the diagonal
    false_negatives = np.sum(confusion, axis=1) - true_positives
    # True Negatives: Total sum of the confusion matrix - (TP + FP + FN)
    true_negatives = np.sum(confusion) - (
      true_positives + false_positives + false_negatives)
    # Specificity for each class
    specificity = true_negatives / (true_negatives + false_positives)
    return specificity
