#!/usr/bin/env python3
""" defines Poisson class that represents Poisson distribution """


class Poisson:
    """
    Represents a Poisson distribution

    class constructor:
        def __init__(self, data=None, lambtha=1.)

    instance attributes:
        lambtha [float]: the expected number of occurances in a given time

    instance methods:
        def pmf(self, k): calculates PMF for given number of successes
        def cdf(self, k): calculates CDF for given number of successes

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
            # If data is provided, ensure it's a list with at least 2 elements
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate lambtha as the average of the data
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the Probality Mass Function for a given number
        of “successes”

        Parameters:
            k (int): number of “successes”

        Returns:
            float: PMF value for k
        """
        # Convert k to an integer if it's not an integer
        if not isinstance(k, int):
            k = int(k)
        # If k is negative, return 0 as PMF
        if k < 0:
            return 0
        # Approximation for e
        e = 2.7182818285
        # Calculate k factorial manually
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i
        # Calculate the PMF using the Poisson formula
        pmf_value = (self.lambtha ** k * e ** -self.lambtha) / factorial
        return pmf_value

    def cdf(self, k):
        """
        Calculates the Cumulative Distribution Function for a given number
        of “successes”

        Parameters:
            k (int): number of “successes”

        Returns:
            float: CDF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        # Calculate the cumulative sum of PMF values from 0 to k
        cumulative_prob = 0
        for i in range(k + 1):
            cumulative_prob += self.pmf(i)
        return cumulative_prob
