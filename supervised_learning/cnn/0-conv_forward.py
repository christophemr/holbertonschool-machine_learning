#!/usr/bin/env python3
"""
Defines a function that performs forward propagation over a convolutional layer
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer.

    Parameters:
    - A_prev (numpy.ndarray): output of the previous layer, shape
    (m, h_prev, w_prev, c_prev)
    - W (numpy.ndarray): kernels for the convolution, shape
    (kh, kw, c_prev, c_new)
    - b (numpy.ndarray): biases applied to the convolution, shape
    (1, 1, 1, c_new)
    - activation (function): activation function applied to the convolution
    - padding (str): either 'same' or 'valid', indicating the
    type of padding used
    - stride (tuple): strides for the convolution, tuple of (sh, sw)

    Returns:
    - output of the convolutional layer (numpy.ndarray)
    """
    # Retrieve dimensions from A_prev's shape and W's shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    # Determine padding based on the 'padding' argument
    if padding == 'same':
        # Calculate padding for 'same' to ensure the output
        # size equals input size
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
        ph = max(ph, 0)  # Ensure non-negative padding
        pw = max(pw, 0)
    elif padding == 'valid':
        ph, pw = (0, 0)

    # Apply padding to A_prev
    A_prev_pad = np.pad(
      A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    # Compute the dimensions of the output
    h_out = (h_prev + 2 * ph - kh) // sh + 1
    w_out = (w_prev + 2 * pw - kw) // sw + 1

    # Initialize the output volume Z
    Z = np.zeros((m, h_out, w_out, c_new))

    # Perform convolution
    for i in range(h_out):
        for j in range(w_out):
            for k in range(c_new):
                # Define the slice from the padded input
                vert_start = i * sh
                vert_end = vert_start + kh
                horiz_start = j * sw
                horiz_end = horiz_start + kw

                # Slice the input and the kernel
                A_slice = (A_prev_pad[
                  :, vert_start:vert_end, horiz_start:horiz_end, :])
                Z[:, i, j, k] = np.sum(A_slice * W[:, :, :, k], axis=(1, 2, 3))

    # Add bias and apply activation function
    Z = Z + b
    A = activation(Z)

    return A
