#!/usr/bin/env python3
"""function that calculates the sum of
the squares of the first n integers
"""


def summation_i_squared(n):
    """
    Calculates the sum of the squares of the first n integers
    Args:
    n (int): The stopping condition (should be a positive integer)
    Returns:
    int: The integer value of the sum of squares of the first n integers
    None: If n is not a valid number
    """
    if not isinstance(n, int) or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
