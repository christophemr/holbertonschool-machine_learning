#!/usr/bin/env python3
"""function that adds 2 matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """returns the sum of 2 matrices element-wise"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    result = []
    for row1, row2 in zip(mat1, mat2):
        result.append([elem1 + elem2 for elem1, elem2 in zip(row1, row2)])
    return result
