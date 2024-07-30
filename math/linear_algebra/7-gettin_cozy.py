#!/usr/bin/env python3
"""function that concatenates 2D matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """returns a concatened new matrix"""
    if axis == 0:
        #check if the number of columns is the same
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        #check if the number of rows is the same
        if len(mat1) != len(mat2):
            return None
        #concatenante the matrices along axis 1(columns)
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        return None
