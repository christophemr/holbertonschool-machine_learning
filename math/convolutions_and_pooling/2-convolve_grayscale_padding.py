#!/usr/bin/env python3
"""
Defines a function that performs convolution with custom padding
on a grayscale image
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding
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
        padding [tuple of (ph, pw)]:
            ph: padding for the height of the image
            pw: padding for the width of the image
    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding  # Get the custom padding for height and width
    # Pad images with zeros using the provided padding
    padded_images = np.pad(
      images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    # Calculate the output dimensions
    out_h = h + 2 * ph - kh + 1
    out_w = w + 2 * pw - kw + 1
    # Prepare the output array
    convoluted = np.zeros((m, out_h, out_w))
    # Perform the convolution
    for i in range(out_h):
        for j in range(out_w):
            # Element-wise multiplication and summation over kernel area
            output = np.sum(
              padded_images[:, i: i + kh, j: j + kw] * kernel, axis=(1, 2))
            convoluted[:, i, j] = output
    return convoluted
