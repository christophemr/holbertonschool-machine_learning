#!/usr/bin/env python3
"""
Performs backpropagation over a convolutional layer of a neural network.
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs backpropagation over a convolutional layer of a neural network.

    Parameters:
    - dZ (numpy.ndarray): gradient of the cost with respect to the output of the conv layer (m, h_new, w_new, c_new)
    - A_prev (numpy.ndarray): output of the previous layer (m, h_prev, w_prev, c_prev)
    - W (numpy.ndarray): weights of the convolution kernels (kh, kw, c_prev, c_new)
    - b (numpy.ndarray): biases (1, 1, 1, c_new)
    - padding (str): either "same" or "valid", specifying the padding type
    - stride (tuple): stride for the convolution (sh, sw)

    Returns:
    - dA_prev: gradient of the cost with respect to the input of the conv layer (m, h_prev, w_prev, c_prev)
    - dW: gradient of the cost with respect to the weights of the conv layer (kh, kw, c_prev, c_new)
    - db: gradient of the cost with respect to the biases of the conv layer (1, 1, 1, c_new)
    """
    # Retrieve dimensions from dZ's shape and other parameters
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    # Initialize gradients for A_prev, W, and b
    dA_prev = np.zeros((m, h_prev, w_prev, c_prev))
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    # Apply padding to A_prev and dA_prev if necessary
    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_new) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_new) // 2 + 1
    else:  # padding == 'valid'
        ph, pw = 0, 0

    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    dA_prev_pad = np.pad(dA_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    # Loop over all training examples
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    # Extract the slice from A_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        # Update the dA_prev after removing padding
        if padding == 'same':
            dA_prev[i] = da_prev_pad[ph:-ph, pw:-pw, :]
        else:
            dA_prev[i] = da_prev_pad

    return dA_prev, dW, db
