#!/usr/bin/env python3
""" defines Binomial class that represents binomial distribution """


class Binomial:
    """
    Represents a binomial distribution
    class constructor:
        def __init__(self, data=None, n=1, p=0.5)

    instance attributes:
        n [int]: the number of Bernoilli trials
        p [float]: the probability of a success

    instance methods:
        def pmf(self, k): calculates PMF for given number of successes
        def cdf(self, k): calculates CDF for given number of successes
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize the binomial distribution.

        Parameters:
            data (list): List of data to estimate the distribution.
            n (int): Number of Bernoulli trials (default=1).
            p (float): Probability of success (default=0.5).
        """
        if data is None:
            # Validate n and p when data is not provided
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            # Validate data when provided
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate n and p from data
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            self.p = 1 - (variance / mean)  # Estimate p
            self.n = round(mean / self.p)  # Estimate n
            self.p = mean / self.n  # Recalculate p

    def factorial(self, num):
        """
        Calculates the factorial of a number manually.

        Parameters:
            num (int): Number to calculate factorial for.

        Returns:
            int: Factorial of num.
        """
        if num == 0 or num == 1:
            return 1
        factorial = 1
        for i in range(2, num + 1):
            factorial *= i
        return factorial

    def pmf(self, k):
        """
        Calculates the PMF for a given number of successes

        Parameters:
            k (int): Number of successes

        Returns:
            float: PMF value for k, or 0 if k is out of range.
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        # Calculate binomial coefficient
        n_choose_k = (self.factorial(self.n) /
                      (self.factorial(k) * self.factorial(self.n - k)))
        # Calculate PMF
        pmf_value = n_choose_k * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return pmf_value
