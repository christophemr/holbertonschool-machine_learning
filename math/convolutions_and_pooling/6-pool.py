#!/usr/bin/env python3
"""
Defines a function that performs pooling on images
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Parameters:
    images (numpy.ndarray): shape (m, h, w, c) containing multiple images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the images
    kernel_shape (tuple): (kh, kw) containing the kernel shape for pooling
        kh is the height of the kernel
        kw is the width of the kernel
    stride (tuple): (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    mode (str): 'max' for max pooling, 'avg' for average pooling

    Returns:
    numpy.ndarray: containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Output dimensions
    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    # Initialize the output array
    pooled = np.zeros((m, out_h, out_w, c))

    # Perform pooling with two loops
    for i in range(out_h):
        for j in range(out_w):
            # Define the current slice for pooling
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            # Slice the image for pooling
            slice_img = (
              images[:, vert_start:vert_end, horiz_start:horiz_end, :])

            # Perform max or average pooling
            if mode == 'max':
                pooled[:, i, j, :] = np.max(slice_img, axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(slice_img, axis=(1, 2))

    return pooled
