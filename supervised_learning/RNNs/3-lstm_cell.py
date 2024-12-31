#!/usr/bin/env python3
"""
Defines the LSTMCell class representing an LSTM unit
"""

import numpy as np


class LSTMCell:
    """
    Represents a Long Short-Term Memory (LSTM) unit
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        Parameters:
        - i: dimensionality of the data
        - h: dimensionality of the hidden state
        - o: dimensionality of the outputs

        Creates public instance attributes:
        - Wf, Wu, Wc, Wo, Wy: weights
        - bf, bu, bc, bo, by: biases
        """
        # Forget gate weights and biases
        self.Wf = np.random.normal(size=(i + h, h))
        self.bf = np.zeros((1, h))

        # Update gate weights and biases
        self.Wu = np.random.normal(size=(i + h, h))
        self.bu = np.zeros((1, h))

        # Intermediate cell state weights and biases
        self.Wc = np.random.normal(size=(i + h, h))
        self.bc = np.zeros((1, h))

        # Output gate weights and biases
        self.Wo = np.random.normal(size=(i + h, h))
        self.bo = np.zeros((1, h))

        # Output weights and biases
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """
        Softmax activation function
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step

        Parameters:
        - h_prev: numpy.ndarray of shape (m, h), the previous hidden state
        - c_prev: numpy.ndarray of shape (m, h), the previous cell state
        - x_t: numpy.ndarray of shape (m, i), the input data for the cell

        Returns:
        - h_next: the next hidden state
        - c_next: the next cell state
        - y: the output of the cell
        """
        # Concatenate input and previous hidden state
        h_x = np.concatenate((h_prev, x_t), axis=1)
        f_gate = self.sigmoid(np.matmul(h_x, self.Wf) + self.bf)
        u_gate = self.sigmoid(np.matmul(h_x, self.Wu) + self.bu)

        c_tilde = np.tanh(np.matmul(h_x, self.Wc) + self.bc)
        c_next = f_gate * c_prev + u_gate * c_tilde

        o_gate = self.sigmoid(np.matmul(h_x, self.Wo) + self.bo)
        h_next = o_gate * np.tanh(c_next)

        # Output
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y
