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

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value

        Parameters:
            x (float): x-value

        Returns:
            float: PDF value for x
        """
        e = 2.7182818285
        pi = 3.1415926536
        coefficient = 1 / (self.stddev * (2 * pi) ** 0.5)
        exponent = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)
        pdf_value = coefficient * (e ** exponent)
        return pdf_value

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value

        Parameters:
            x (float): x-value

        Returns:
            float: CDF value for x
        """
        pi = 3.1415926536
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        # Approximate erf(z)
        erf_z = (2 / (pi ** 0.5)) * (z - (z ** 3) / 3 + (z ** 5) / 10 -
                                     (z ** 7) / 42 + (z ** 9) / 216)
        # Calculate CDF using the erf approximation
        cdf_value = 0.5 * (1 + erf_z)
        return cdf_value
