#!/usr/bin/env python3
""" defines Exponential class that represents exponential distribution """


class Exponential:
    """
    class that represents exponential distribution

    class constructor:
        def __init__(self, data=None, lambtha=1.)

    instance attributes:
        lambtha [float]: the expected number of occurances in a given time

    instance methods:
        def pdf(self, x): calculates PDF for given time period
        def cdf(self, x): calculates CDF for given time period
    """

    def __init__(self, data=None, lambtha=1.):
        """
        class constructor

        parameters:
            data [list]: data to be used to estimate the distibution
            lambtha [float]: the expected number of occurances on a given time

        Sets the instance attribute lambtha as a float
        If data is not given:
            Use the given lambtha or
            raise ValueError if lambtha is not positive value
        If data is given:
            Calculate the lambtha of data
            Raise TypeError if data is not a list
            Raise ValueError if data does not contain at least two data points
        """
        # If data is provided, calculate lambtha based on data
        if data is None:
            # If data is not provided, use the given lambtha value
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # If data is provided, ensure it is a list with at least 2ch elements
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate lambtha as the inverse of the mean of the data
            self.lambtha = 1 / (sum(data) / len(data))
