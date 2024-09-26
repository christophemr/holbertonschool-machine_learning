#!/usr/bin/env python3
"""
Defines a function that performs forward propagation over a pooling layer
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer.

    Parameters:
    - A_prev (numpy.ndarray): output of the previous layer, shape
    (m, h_prev, w_prev, c_prev)
    - kernel_shape (tuple): size of the pooling kernel (kh, kw)
    - stride (tuple): strides for the pooling, tuple of (sh, sw)
    - mode (str): either 'max' or 'avg', indicating the type of pooling

    Returns:
    - output of the pooling layer (numpy.ndarray)
    """
    # Retrieve dimensions from A_prev's shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Compute the dimensions of the output
    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1

    # Initialize the output volume
    output = np.zeros((m, h_out, w_out, c_prev))

    # Perform pooling operation
    for i in range(h_out):
        for j in range(w_out):
            # Define the slice from the input
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            # Slice the input volume for the current window
            A_slice = A_prev[:, vert_start:vert_end, horiz_start:horiz_end, :]

            if mode == 'max':
                output[:, i, j, :] = np.max(A_slice, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(A_slice, axis=(1, 2))

    return output
