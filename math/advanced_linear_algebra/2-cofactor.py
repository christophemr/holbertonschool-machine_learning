#!/usr/bin/env python3
"""
Function that calculates the cofactor matrix of a matrix
"""


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix

    Parameters:
        matrix [list of lists]: matrix whose cofactor matrix should be
                                calculated

    Returns:
        The cofactor matrix of matrix
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

    # Helper function to calculate the determinant of a matrix
    def determinant(sub_matrix):
        if len(sub_matrix) == 1:
            return sub_matrix[0][0]
        if len(sub_matrix) == 2:
            return (sub_matrix[0][0] * sub_matrix[1][1] -
                    sub_matrix[0][1] * sub_matrix[1][0])
        det = 0
        for col in range(len(sub_matrix)):
            minor = [row[:col] + row[col+1:] for row in sub_matrix[1:]]
            det += ((-1) ** col) * sub_matrix[0][col] * determinant(minor)
        return det

    size = len(matrix)
    cofactor_matrix = []

    for i in range(size):
        row_cofactor = []
        for j in range(size):
            # Create the minor matrix by excluding the i-th row and j-th column
            minor_matrix = [
                row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])
            ]
            # Calculate the cofactor by applying the checkerboard pattern
            cofactor_value = ((-1) ** (i + j)) * determinant(minor_matrix)
            row_cofactor.append(cofactor_value)
        cofactor_matrix.append(row_cofactor)

    return cofactor_matrix
