#!/usr/bin/env python3
"""
Defines a function that performs same convolution
on a grayscale image
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images

    Parameters:
        images [numpy.ndarray of shape (m, h, w)]:
            contains multiple grayscale images
            m: number of images
            h: height in pixels of all images
            w: width in pixels of all images
        kernel [numpy.ndarray of shape (kh, kw)]:
            contains the kernel for the convolution
            kh: height of the kernel
            kw: width of the kernel

    Returns:
        numpy.ndarray containing convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding for both odd and even kernel sizes
    if (kh % 2) == 1:
        ph = (kh - 1) // 2
    else:
        ph = kh // 2

    if (kw % 2) == 1:
        pw = (kw - 1) // 2
    else:
        pw = kw // 2

    # Pad images with zeros
    padded_images = np.pad(
      images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # Prepare the output array with the same shape as the original images
    convoluted = np.zeros((m, h, w))

    # Perform the convolution
    for i in range(h):
        for j in range(w):
            # Element-wise multiplication and summation over kernel area
            output = np.sum(
              padded_images[:, i: i + kh, j: j + kw] * kernel, axis=(1, 2))
            convoluted[:, i, j] = output

    return convoluted
