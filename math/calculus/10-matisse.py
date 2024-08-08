#!/usr/bin/env python3
"""function that calculates the derivative
of a polynomial
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial

    Args:
    poly (list): List of coefficients representing a polynomial

    Returns:
    list: A new list of coefficients representing
    the derivative of the polynomial
    None: If poly is not valid
    """
    if (not isinstance(poly, list) or
       not all(isinstance(coef, (int, float)) for coef in poly)
       or len(poly) == 0):
        return None

    if len(poly) == 1:
        return [0]

    derivative = [i * poly[i] for i in range(1, len(poly))]

    if all(coef == 0 for coef in derivative):
        return [0]

    return derivative
