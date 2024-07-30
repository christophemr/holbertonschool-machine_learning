#!/usr/bin/env python3
"""function that returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """returns a new matrix (transpose of a 2D matrix)"""
    return [list(row) for row in zip(*matrix)]
