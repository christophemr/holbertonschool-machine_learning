#!/usr/bin/env python3
"""this module will define a binary image
    classifier from scratch using numpy
"""

import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Performs one pass of gradient descent
        to update the weights and bias.
        Parameters:
        -----------
        X : numpy.ndarray
            The input data array of shape (nx, m),
            where nx is the number of input features
            and m is the number of examples.
        Y : numpy.ndarray
            Correct labels for the input data with shape (1, m),
            where m is the number of examples.
        A : numpy.ndarray
            Activated output of the neuron for each example with shape (1, m).
        alpha : float, optional (default=0.05)
            The learning rate used to update the weights and bias.
        Updates:
        --------
        __W : numpy.ndarray
            The weights vector after applying gradient descent.
        __b : float
            The bias after applying gradient descent.
        """
        # Number of examples
        m = X.shape[1]
        # Compute the gradient of the loss with respect to Z (dZ)
        dZ = A - Y
        # Compute the gradient of the loss with respect to W (dW)
        dW = np.matmul(dZ, X.T) / m
        # Compute the gradient of the loss with respect to b (db)
        db = np.sum(dZ) / m
        # Update the weights using gradient descent
        self.__W = self.__W - alpha * dW
        # Update the bias using gradient descent
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the neuron over a specified number of
        iterations using gradient descent.

        Parameters:
        -----------
        X : numpy.ndarray
            The input data array of shape (nx, m),
            where nx is the number of input features
            and m is the number of examples.
        Y : numpy.ndarray
            Correct labels for the input data with shape (1, m),
            where m is the number of examples.
        iterations : int, optional (default=5000)
            The number of iterations to train over.
        alpha : float, optional (default=0.05)
            The learning rate used to update the weights and bias.
        verbose : bool, optional (default=True)
            Whether to print training progress information.
        graph : bool, optional (default=True)
            Whether to plot the training cost over time.
        step : int, optional (default=100)
            The number of iterations between printing
            and plotting training progress.

        Raises:
        -------
        TypeError:
            If `iterations` is not an integer.
            If `alpha` is not a float.
            If `step` is not an integer.
        ValueError:
            If `iterations` is not positive.
            If `alpha` is not positive.
            If `step` is not positive or is greater than `iterations`.

        Returns:
        --------
        tuple : (numpy.ndarray, float)
            - The predicted labels for each example
            (1 if the activated output >= 0.5, else 0).
            - The cost of the model after training.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        # Initialize a list to store the costs for plotting
        costs = []

        for i in range(iterations + 1):
            A = self.forward_prop(X)
            cost = self.cost(Y, A)
            # Save cost for plotting
            if graph:
                costs.append(cost)
            # Print progress if verbose
            if verbose and i % step == 0:
                print(f"Cost after {i} iterations: {cost}")
            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)
        # Plot the training cost if graph is True
        if graph:
            plt.plot(np.arange(0, iterations + 1), costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)