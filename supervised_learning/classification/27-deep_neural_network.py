#!/usr/bin/env python3
"""
defines the DeepNeuralNetwork class that
performs multiclass classification
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification.
    """

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network.
        Parameters:
        -----------
        nx : int
            Number of input features
        layers : list
            A list containing the number of neurons in each layer
        Raises:
        -------
        TypeError:
            If nx is not an integer or if layers is not
            a list of positive integers.
        ValueError:
            If nx is less than 1 or if layers is an empty list.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # Initialize the number of layers, cache, and weights dictionary
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialize layers and validate
        for layer_index in range(self.__L):
            layer_size = layers[layer_index]
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise TypeError("layers must be a list of positive integers")

            prev_layer_size = (nx if layer_index == 0 else
                               layers[layer_index - 1])

            # He initialization: random weights scaled by sqrt
            # (2 / number of inputs)
            self.__weights['W' + str(layer_index + 1)] = (
                np.random.randn(layer_size, prev_layer_size)
                * np.sqrt(2 / prev_layer_size)
            )
            # column vector with a size equal to the number
            # of neurons in this layer
            self.__weights['b' + str(layer_index + 1)] = (
              np.zeros((layer_size, 1))
              )

    @property
    def L(self):
        """Getter for the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for cache (intermediary values)."""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights (weights and biases)."""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (nx, m) where nx is the number
            of input features and m is the number of examples.
        Returns:
        --------
        A : numpy.ndarray
            The output of the neural network after forward propagation.
        cache : dict
            Dictionary containing all the intermediary
            activations in the network.
        """
        self.__cache['A0'] = X

        # Perform forward propagation through each layer
        for layer_index in range(self.__L):
            W = self.__weights['W' + str(layer_index + 1)]
            b = self.__weights['b' + str(layer_index + 1)]
            A_prev = self.__cache['A' + str(layer_index)]
            Z = np.matmul(W, A_prev) + b  # Linear combination
            # output layer uses softmax
            if layer_index == self.__L - 1:
                Z = Z - np.max(Z, axis=0, keepdims=True)
                A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
            else:
                A = 1 / (1 + np.exp(-Z))  # Hidden layer uses Sigmoid
            # Store the activated output in the cache
            self.__cache['A' + str(layer_index + 1)] = A
        # The final layer's output is the output of the neural network
        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        Parameters:
        -----------
        Y : numpy.ndarray
            Shape (1, m) that contains the correct labels for the input data.
        A : numpy.ndarray
            Shape (1, m) containing the activated output
            of the neuron for each example.
        Returns:
        --------
        cost : float
            The cost of the model.
        """
        # number of examples
        m = Y.shape[1]
        # log loss(categorical)
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.
        Parameters:
        -----------
        X : numpy.ndarray
            Shape (nx, m) that contains the input data.
        Y : numpy.ndarray
            Shape (1, m) that contains the correct labels 4 the input data.
        Returns:
        --------
        prediction : numpy.ndarray
            Shape (1, m) containing the predicted labels 4 each example.
        cost : float
            The cost of the network.
        """
        A, _ = self.forward_prop(X)
        predictions = np.eye(A.shape[0])[np.argmax(A, axis=0)].T
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.

        Parameters:
        -----------
        Y : numpy.ndarray
            Shape (1, m) that contains the correct labels 4 the input data.
        cache : dict
            Dictionary containing all the intermediary values of the network.
        alpha : float
            The learning rate.
        Updates:
        --------
        Updates the private attribute __weights.
        """
        m = Y.shape[1]
        A_last = cache['A' + str(self.__L)]
        delta_Z = A_last - Y

        for layer_index in reversed(range(1, self.__L + 1)):
            A_prev = cache['A' + str(layer_index - 1)]
            W = self.__weights['W' + str(layer_index)]

            dW = np.matmul(delta_Z, A_prev.T) / m
            db = np.sum(delta_Z, axis=1, keepdims=True) / m
            dA_prev = np.matmul(W.T, delta_Z)
            # Update weights and biases
            self.__weights['W' + str(layer_index)] -= alpha * dW
            self.__weights['b' + str(layer_index)] -= alpha * db
            # If not the first layer, calculate delta_Z 4 the next layer
            if layer_index > 1:
                delta_Z = dA_prev * A_prev * (1 - A_prev)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network by updating the weights and biases.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (nx, m).
        Y : numpy.ndarray
            Labels of shape (1, m).
        iterations : int
            Number of iterations to train over.
        alpha : float
            Learning rate.
        verbose : bool
            If True, print the cost after every 'step' iterations.
        graph : bool
            If True, graph the cost during training.
        step : int
            Interval of steps at which to print or graph the cost.

        Returns:
        --------
        prediction : numpy.ndarray
            Prediction after the final iteration.
        cost : float
            Cost of the network after the final iteration.

        Raises:
        -------
        TypeError:
            If iterations is not an integer or alpha is not a float.
        ValueError:
            If iterations is not positive or alpha is not positive.
        TypeError:
            If step is not an integer.
        ValueError:
            If step is not positive and less than or equal to iterations.
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

        for i in range(iterations):
            # Forward propagation
            A, _ = self.forward_prop(X)

            # Gradient descent
            self.gradient_descent(Y, self.__cache, alpha)
            # Record and print cost at intervals
            if i % step == 0 or i == iterations - 1:
                cost = self.cost(Y, A)
                costs.append(cost)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

        # Plot cost graph if required
        if graph:
            plt.plot(range(0, iterations + 1, step), costs)
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title('Training Cost')
            plt.show()

        # After training, evaluate the model
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format.

        Parameters:
        -----------
        filename : str
            The file to which the object should be saved.
            If filename does not have the extension .pkl, it is added.
        """
        # Add the .pkl extension if it's not there
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    # Add the load static method
    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object from a file.
        Parameters:
        -----------
        filename : str
            The file from which the object should be loaded.
        Returns:
        --------
        DeepNeuralNetwork or None
            The loaded object, or None if the file doesn't exist.
        """
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        return None