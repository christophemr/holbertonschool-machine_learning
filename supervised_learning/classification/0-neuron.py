#!/usr/bin/env python3
"""this module will define a binary image
    classifier from scratch using numpy
"""

import numpy as np


class Neuron:
    """
    Class that defines a single neuron performing binary classification.
    public instance attributes:
    -----------
    W : numpy.ndarray
        The weights vector for the neuron. Upon instantiation,
        it is initialized using a random normal distribution.
    b : float
        The bias for the neuron. Upon instantiation, it is initialized to 0.
    A : float
        The activated output of the neuron (prediction).
        Upon instantiation, it is initialized to 0.
    """
    def __init__(self, nx):
        """Constructor for the Neuron class
        Parameters:
    -----------
    nx : int
        The number of input features to the neuron.
    Raises:
    -------
    TypeError:
        If `nx` is not an integer.
    ValueError:
        If `nx` is less than 1.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
