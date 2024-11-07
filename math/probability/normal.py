#!/usr/bin/env python3
""" defines Normal class that represents normal distribution """


class Normal:
    """
    class that represents normal distribution

    class constructor:
        def __init__(self, data=None, mean=0., stddev=1.)

    instance attributes:
        mean [float]: the mean of the distribution
        stddev [float]: the standard deviation of the distribution

    instance methods:
        def z_score(self, x): calculates the z-score of a given x-value
        def x_value(self, z): calculates the x-value of a given z-score
        def pdf(self, x): calculates PDF for given x-value
        def cdf(self, x): calculates CDF for given x-value
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        class constructor

        parameters:
            data [list]: data to be used to estimate the distibution
            mean [float]: the mean of the distribution
            stddev [float]: the standard deviation of the distribution
        """
        # If data is not provided, use given mean and stddev
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            # Validate data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate mean
            self.mean = float(sum(data) / len(data))
            # Calculate standard deviation
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value

        Parameters:
            x (float): x-value

        Returns:
            float: z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score

        Parameters:
            z (float): z-score

        Returns:
            float: x-value corresponding to z
        """
        return z * self.stddev + self.mean
