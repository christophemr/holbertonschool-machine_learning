#!/usr/bin/env python3
"""this module will define a binary image
    classifier from scratch using numpy
"""

import numpy as np


class Neuron:
    """
    Class that defines a single neuron performing binary classification.
    Private instance attributes:
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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """gets the private instance attribute __W
        __W is the weights vector for the neuron
        """
        return (self.__W)

    @property
    def b(self):
        """gets the private instance attribute __b
        __b is the bias for the neuron
        """
        return (self.__b)

    @property
    def A(self):
        """gets the private instance attribute __A
        __A is the activated output of the neuron
        """
        return (self.__A)

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        using the sigmoid activation function.

        Parameters:
        -----------
        X : numpy.ndarray
            The input data array of shape (nx, m),
            where nx is the number of input features
            and m is the number of examples.

        Returns:
        --------
        __A : numpy.ndarray
            The activated output of the neuron after
            applying the sigmoid function.
        """
        # Compute the linear combination of inputs, weights,
        # and bias using numpy.matmul
        Z = np.matmul(self.__W, X) + self.__b
        # Apply the sigmoid activation function
        self.__A = 1 / (1 + np.exp(-Z))
        # Return the activated output
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        Parameters:
        -----------
        Y : numpy.ndarray
            Correct labels for the input data with shape (1, m),
            where m is the number of examples.
        A : numpy.ndarray
            Activated output of the neuron for each example with shape (1, m).
        Returns:
        --------
        cost : float
            The cost (logistic regression loss) of the model.
        """
        # Number of examples
        m = Y.shape[1]

        # Compute the cost using the cross-entropy loss formula
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions and returns
        the predicted labels and the cost.
        Parameters:
        -----------
        X : numpy.ndarray
            The input data array of shape (nx, m),
            where nx is the number of input features
            and m is the number of examples.
        Y : numpy.ndarray
            Correct labels for the input data with shape (1, m),
            where m is the number of examples.
        Returns:
        --------
        tuple : (numpy.ndarray, float)
            - The predicted labels for each example
            (1 if the activated output >= 0.5, else 0).
            - The cost of the model.
        """
        # Perform forward propagation to get the activated output
        A = self.forward_prop(X)
        # Compute predictions: 1 if A >= 0.5, else 0
        predictions = np.where(A >= 0.5, 1, 0)
        # Calculate the cost
        cost = self.cost(Y, A)
        return predictions, cost
