#!/usr/bin/env python3
"""function that concatenates two matrices along a specific axis"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """returns a ndarray (concatenation) of 2 matrices"""
    return np.concatenate((mat1, mat2), axis=axis)
