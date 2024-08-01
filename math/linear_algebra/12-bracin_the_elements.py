#!/usr/bin/env python3
"""function that performs elementary operations"""


def np_elementwise(mat1, mat2):
    """performs element-wise addition, subtraction, multiplication
    , and division on 2matrices
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return (add, sub, mul, div)
