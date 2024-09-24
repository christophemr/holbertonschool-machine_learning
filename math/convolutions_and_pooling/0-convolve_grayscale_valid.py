#!/usr/bin/env python3
"""
Defines a function that performs valid convolution
on a grayscale image
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.
    Parameters:
        images (numpy.ndarray): 3D array of shape (m, h, w)
        containing multiple grayscale images.
        kernel (numpy.ndarray): 2D array of shape (kh, kw)
        representing the kernel/filter.
    Returns:
        numpy.ndarray: 3D array of shape (m, output_h, output_w)
        containing the convolved images.
    """
    # Get the dimensions of images and the kernel
    m, h, w = images.shape  # m = number of images, h = height, w = width
    kh, kw = kernel.shape   # kh = kernel height, kw = kernel width
    # Compute the dimensions of the output after valid convolution
    output_h = h - kh + 1
    output_w = w - kw + 1
    # Initialize the output array
    output = np.zeros((m, output_h, output_w))
    # Perform convolution for each image
    for img in range(m):
        for i in range(output_h):
            for j in range(output_w):
                # Perform element-wise multiplication and sum the result
                output[img, i, j] = np.sum(
                    images[img, i:i+kh, j:j+kw] * kernel)
    return output
