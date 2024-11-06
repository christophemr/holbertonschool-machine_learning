#!/usr/bin/env python3
"""
Function that calculates the minor matrix of a matrix
"""


def minor(matrix):
    """
    Calculates the minor matrix of a matrix

    Parameters:
        matrix [list of lists]: matrix whose minor matrix should be calculated

    Returns:
        The minor matrix of matrix
    """
    # Input validation
    if (not isinstance(matrix, list) or
            not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Base case for a 1x1 matrix
    if len(matrix) == 1:
        return [[1]]

    # Calculate minor matrix
    def determinant(sub_matrix):
        """ Helper function to calculate the determinant of a matrix """
        if len(sub_matrix) == 1:
            return sub_matrix[0][0]
        if len(sub_matrix) == 2:
            return sub_matrix[0][0] * sub_matrix[1][1]  \
                  - sub_matrix[0][1] * sub_matrix[1][0]
        det = 0
        for col in range(len(sub_matrix)):
            minor = [row[:col] + row[col+1:] for row in sub_matrix[1:]]
            det += ((-1) ** col) * sub_matrix[0][col] * determinant(minor)
        return det

    size = len(matrix)
    minor_matrix = []

    for i in range(size):
        row_minor = []
        for j in range(size):
            # Create the sub-matrix by excluding the i-th row and j-th column
            sub_matrix = [row[:j] + row[j+1:] for row in (
                matrix[:i] + matrix[i+1:])]
            # Calculate determinant of the sub-matrix
            row_minor.append(determinant(sub_matrix))
        minor_matrix.append(row_minor)

    return minor_matrix
