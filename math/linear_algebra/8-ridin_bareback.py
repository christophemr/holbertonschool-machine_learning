#!/usr/bin/env python3
"""function that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """returns a new matrix product of 2D matrices"""
    # check if number of columns is equal to number of rows
    if len(mat1[0]) != len(mat2):
        return None
    # initialize the zeros matrix
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    # matrix multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result
