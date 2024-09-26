#!/usr/bin/env python3
"""
Defines a function that performs back propagation
over a convolutional layer of a neural network
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.
    Parameters:
    - dZ (numpy.ndarray): partial derivatives with respect to the unactivated
      output of the convolutional layer (m, h_new, w_new, c_new)
    - A_prev (numpy.ndarray): output of the previous layer
    (m, h_prev, w_prev, c_prev)
    - W (numpy.ndarray): kernels for the convolution (kh, kw, c_prev, c_new)
    - b (numpy.ndarray): biases applied to the convolution (1, 1, 1, c_new)
    - padding (str): either 'same' or 'valid', indicating the type of padding
    used
    - stride (tuple): (sh, sw) containing the strides for the convolution

    Returns:
    - dA_prev (numpy.ndarray): partial derivatives with respect
    to the previous layer
    - dW (numpy.ndarray): partial derivatives with respect to the kernels
    - db (numpy.ndarray): partial derivatives with respect to the biases
    """
    # Retrieve dimensions from dZ's shape
    m, h_new, w_new, c_new = dZ.shape
    h_prev, w_prev, c_prev = A_prev.shape[1:]

    kh, kw, _, _ = W.shape
    sh, sw = stride

    # Initialize derivatives for dA_prev, dW, and db
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    # Pad A_prev and dA_prev if necessary
    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    else:
        ph, pw = 0, 0

    A_prev_pad = np.pad(
      A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
      mode='constant', constant_values=0)
    dA_prev_pad = np.pad(
      dA_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
      mode='constant', constant_values=0)

    # Compute db: sum over the training examples, height, and width
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Loop over the training examples
    for i in range(m):
        # Select the ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        # Loop over the height and width of the output
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    # Find the corners of the current slice
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    # Get the slice of the padded input image
                    a_slice = (
                      a_prev_pad[vert_start:vert_end, horiz_start:horiz_end,
                                 :])

                    # Update the gradients for the slice, kernel, and input
                    (da_prev_pad[vert_start:vert_end, horiz_start:horiz_end,
                                 :]) += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

        # Set the ith training example's dA_prev (remove padding if applied)
        if padding == "same":
            dA_prev[i, :, :, :] = da_prev_pad[ph:-ph, pw:-pw, :]
        else:
            dA_prev[i, :, :, :] = da_prev_pad

    return dA_prev, dW, db
