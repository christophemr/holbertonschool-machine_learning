#!/usr/bin/env python3
"""
Defines function that calculates the determinant of a matrix
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix

    parameters:
        matrix [list of lists]:
            matrix whose determinant should be calculated

    returns:
        the determinant of matrix
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    matrix_size = len(matrix)
    if matrix_size == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(row) == 0 and matrix_size == 1:
            return 1
        if len(row) != matrix_size:
            raise ValueError("matrix must be a square matrix")
    if matrix_size == 1:
        return matrix[0][0]
    if matrix_size == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        return (a * d) - (b * c)

    # Recursive calculation for larger matrices
    determinant_value = 0
    sign = 1
    for col in range(matrix_size):
        pivot_element = matrix[0][col]
        minor_matrix = []
        for row in range(1, matrix_size):
            new_row = [matrix[row][c] for c in range(matrix_size) if c != col]
            minor_matrix.append(new_row)
        determinant_value += pivot_element * sign * determinant(minor_matrix)
        sign *= -1
    return determinant_value
