#!/usr/bin/env python3
"""
Function that calculates the adjugate matrix of a matrix
"""


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix

    Parameters:
        matrix [list of lists]: matrix whose adjugate matrix should be
                                calculated

    Returns:
        The adjugate matrix of matrix
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

    # Helper function to calculate the cofactor matrix
    def cofactor(matrix):
        size = len(matrix)
        cofactor_matrix = []

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

        for i in range(size):
            row_cofactor = []
            for j in range(size):
                minor_matrix = [
                    row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])
                ]
                cofactor_value = ((-1) ** (i + j)) * determinant(minor_matrix)
                row_cofactor.append(cofactor_value)
            cofactor_matrix.append(row_cofactor)

        return cofactor_matrix

    # Calculate the cofactor matrix
    cofactor_matrix = cofactor(matrix)

    # Transpose the cofactor matrix to get the adjugate matrix
    adjugate_matrix = [[cofactor_matrix[j][i] for j in range(len(matrix))]
                       for i in range(len(matrix))]

    return adjugate_matrix
