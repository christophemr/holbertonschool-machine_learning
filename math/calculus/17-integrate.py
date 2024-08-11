#!usr/bin/env python3
"""function that calculates the integral of
a plynomial
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial

    Parameters:
        poly (list): List of coefficients representing a polynomial.
            The index of the list represents the power of x
            the coefficient belongs to.
        C (int): The integration constant.

    Returns:
        list: A new list of coefficients representing
        the integral of the polynomial.
        None: If poly or C are not valid.
    """
    if (not isinstance(poly, list) or
       not all(isinstance(coef, (int, float)) for coef in poly) or
       not isinstance(C, int)):
        return None

    # Compute the integral coefficients by dividing
    # each by its new power (index + 1)
    integral = [C] + [coef / (i + 1) for i, coef in enumerate(poly)]

    # Convert any whole numbers (float that represent integers) to integers
    integral = [int(coef) if isinstance(coef, float) and
                coef.is_integer() else coef for coef in integral]

    # Remove trailing zeros
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
