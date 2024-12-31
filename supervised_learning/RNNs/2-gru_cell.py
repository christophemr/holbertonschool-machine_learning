#!/usr/bin/env python3
"""
Defines the GRUCell class representing a gated recurrent unit (GRU)
"""

import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit (GRU) cell
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        Parameters:
        - i: dimensionality of the data
        - h: dimensionality of the hidden state
        - o: dimensionality of the outputs

        Creates public instance attributes:
        - Wz, Wr, Wh, Wy: weights
        - bz, br, bh, by: biases
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros((1, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros((1, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))
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

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Parameters:
        - h_prev: numpy.ndarray of shape (m, h), the previous hidden state
        - x_t: numpy.ndarray of shape (m, i), the input data for the cell

        Returns:
        - h_next: the next hidden state
        - y: the output of the cell
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        z_gate = self.sigmoid(np.matmul(h_x, self.Wz) + self.bz)

        r_gate = self.sigmoid(np.matmul(h_x, self.Wr) + self.br)

        r_h_x = np.concatenate((r_gate * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(r_h_x, self.Wh) + self.bh)
        h_next = (1 - z_gate) * h_prev + z_gate * h_tilde

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
