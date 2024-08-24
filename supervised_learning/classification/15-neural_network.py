#!/usr/bin/env python3
"""This module defines a neural network with one hidden
layer performing binary classification."""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    Class that defines a neural network with one hidden
    layer performing binary classification.
    Attributes:
    -----------
    __W1 : numpy.ndarray
        The weights vector 4 the hidden layer.
        Initialized using a random normal distribution.
    __b1 : numpy.ndarray
        The bias 4 the hidden layer. Initialized with 0’s.
    __A1 : float
        The activated output 4 the hidden layer. Initialized to 0.
    __W2 : numpy.ndarray
        The weights vector 4 the output neuron.
        Initialized using a random normal distribution.
    __b2 : float
        The bias 4 the output neuron. Initialized to 0.
    __A2 : float
        The activated output 4 the output neuron
        (prediction). Initialized to 0.
    Methods:
    --------
    __init__(self, nx, nodes):
        Constructor 4 the NeuralNetwork class.
    Getter methods:
    ---------------
    W1(self): Returns the weights vector 4 the hidden layer.
    b1(self): Returns the bias 4 the hidden layer.
    A1(self): Returns the activated output 4 the hidden layer.
    W2(self): Returns the weights vector 4 the output neuron.
    b2(self): Returns the bias 4 the output neuron.
    A2(self): Returns the activated output 4 the output neuron.
    """

    def __init__(self, nx, nodes):
        """Constructor 4 the NeuralNetwork class
        Parameters:
        -----------
        nx : int
            The number of input features to the neural network.
        nodes : int
            The number of nodes found in the hidden layer.
        Raises:
        -------
        TypeError:
            If `nx` is not an integer.
            If `nodes` is not an integer.
        ValueError:
            If `nx` is less than 1.
            If `nodes` is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        # Initialize weights and biases 4 the hidden layer
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Initialize weights and biases 4 the output layer
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter method 4 the weights vector of the hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """Getter method 4 the bias of the hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """Getter method 4 the activated output of the hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """Getter method 4 the weights vector of the output neuron."""
        return self.__W2

    @property
    def b2(self):
        """Getter method 4 the bias of the output neuron."""
        return self.__b2

    @property
    def A2(self):
        """Getter method 4 the activated output of
        the output neuron (prediction)."""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.
        Parameters:
        -----------
        X : numpy.ndarray
            The input data array of shape (nx, m),
            where nx is the number of input features
            and m is the number of examples.
        Returns:
        --------
        tuple : (__A1, __A2)
            - __A1: The activated output of the hidden layer.
            - __A2: The activated output of the output neuron (prediction).
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

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
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions and
        returns the predicted labels and cost.
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
        self.forward_prop(X)
        # Compute predictions: 1 if A >= 0.5, else 0
        predictions = np.where(self.__A2 >= 0.5, 1, 0)
        # Calculate the cost
        cost = self.cost(Y, self.__A2)
        return predictions, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Performs one pass of gradient descent on the neural network,
        updating weights and biases.
        Parameters:
        -----------
        X : numpy.ndarray
            The input data array of shape (nx, m),
            where nx is the number of input features
            and m is the number of examples.
        Y : numpy.ndarray
            Correct labels for the input data with shape (1, m),
            where m is the number of examples.
        A1 : numpy.ndarray
            The activated output of the hidden layer with shape (nodes, m).
        A2 : numpy.ndarray
            The activated output of the output neuron with shape (1, m).
        alpha : float, optional (default=0.05)
            The learning rate used to update the weights and biases.
        Updates:
        --------
        __W1 : numpy.ndarray
            The weights vector for the hidden layer.
        __b1 : numpy.ndarray
            The bias for the hidden layer.
        __W2 : numpy.ndarray
            The weights vector for the output neuron.
        __b2 : numpy.ndarray
            The bias for the output neuron.
        """
        m = Y.shape[1]
        # Calculate the gradient of the cost with respect to Z2
        # (output layer)
        dZ2 = A2 - Y
        # Calculate the gradients with respect to W2 and b2
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        # Calculate the gradient of the cost with respect to A1
        dA1 = np.matmul(self.__W2.T, dZ2)
        # Calculate the gradient of the cost with respect to Z1
        # (hidden layer)
        dZ1 = dA1 * A1 * (1 - A1)
        # Calculate the gradients with respect to W1 and b1
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        # Update the weights and biases
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network.
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
            The number of iterations to train the neural network.
        alpha : float, optional (default=0.05)
            The learning rate used to update the weights and biases.
        Returns:
        --------
        tuple : (numpy.ndarray, float)
            - The predicted labels for the training data
            after the final iteration.
            - The cost of the model after the final iteration.
        Raises:
        -------
        TypeError:
            If `iterations` is not an integer.
            If `alpha` is not a float.
        ValueError:
            If `iterations` is not positive.
            If `alpha` is not positive.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            # Forward pass
            A1, A2 = self.forward_prop(X)
            # Gradient descent
            self.gradient_descent(X, Y, A1, A2, alpha)
        # Return the final evaluation after all iterations
        return self.evaluate(X, Y)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the neural network.
        Parameters
        -----------
        X : numpy.ndarray
            Input data of shape (nx, m), where nx is the number
            of features and m is the number of examples.
        Y : numpy.ndarray
            Correct labels for the input data of shape (1, m).
        iterations : int, optional
            Number of iterations to train over (default is 5000).
        alpha : float, optional
            Learning rate (default is 0.05).
        verbose : bool, optional
            If True, prints the cost every 'step' iterations
            (default is True).
        graph : bool, optional
            If True, plots the cost over iterations after
            training (default is True).
        step : int, optional
            Steps in which to print or plot the cost (default is 100).
        Returns:
        --------
        tuple
            The evaluation of the training data after 'iterations'
            of training, containing the prediction and the cost.
        Raises:
        -------
        TypeError:
            If 'iterations' is not an integer.
            If 'alpha' is not a float.
            If 'step' is not an integer (when verbose or graph is True).
        ValueError:
            If 'iterations' is not positive.
            If 'alpha' is not positive.
            If 'step' is not positive or greater than 'iterations'
            (when verbose or graph is True).
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []
        steps = []
        for i in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A2)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                if graph:
                    costs.append(cost)
                    steps.append(i)
            if i < iterations:
                self.gradient_descent(X, Y, A1, A2, alpha)
        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
