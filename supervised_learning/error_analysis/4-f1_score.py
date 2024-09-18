#!/usr/bin/env python3
"""
function that calculates the F1 score
for each class in a confusion matrix
"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels.

    Returns:
        numpy.ndarray: F1 score for each class, shape (classes,)
    """
    # Calculate sensitivity (recall) for each class
    recall = sensitivity(confusion)
    # Calculate precision for each class
    prec = precision(confusion)
    # Calculate the F1 score using the formula
    f1 = 2 * (prec * recall) / (prec + recall)
    return f1
