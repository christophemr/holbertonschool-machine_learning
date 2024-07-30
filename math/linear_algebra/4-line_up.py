#!/usr/bin/env python3
"""function that adds 2 arrays element-wise"""


def add_arrays(arr1, arr2):
    """returns the sum of 2 arrayselement-wise in the form of a list"""
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
