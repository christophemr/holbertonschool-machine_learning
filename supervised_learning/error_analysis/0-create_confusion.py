#!/usr/bin/env python3
"""
function that creates a confusion matrix
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Parameters:
    labels (numpy.ndarray): One-hot encoded true labels
    of shape (m, classes).
    logits (numpy.ndarray): One-hot encoded predicted
    labels of shape (m, classes).

    Returns:
    numpy.ndarray: Confusion matrix of shape (classes, classes).
    """
    # Convert one-hot encoded labels and logits to class indices
    true_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)
    # Number of classes
    num_classes = labels.shape[1]
    # Initialize the confusion matrix with zeros
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=float)
    # Fill the confusion matrix
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[true_label, predicted_label] += 1
    return confusion_matrix
