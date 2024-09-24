#!/usr/bin/env python3
"""
Defines a function that performs convolution
on a grayscale image with given padding and stride.
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with
    custom padding and stride
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
        padding [tuple of (ph, pw), 'same', or 'valid']:
            padding type or specific padding to apply
        stride [tuple of (sh, sw)]:
            sh: stride for height of the image
            sw: stride for width of the image
    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    # Determine padding based on the 'padding' argument
    if padding == 'same':
        # Calculate padding for 'same' to ensure the output size
        # equals input size
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = (0, 0)
    else:
        ph, pw = padding  # Custom padding provided as a tuple
    # Apply padding to the input images
    padded_images = np.pad(
      images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    # Compute the output dimensions
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    # Prepare the output array
    convoluted = np.zeros((m, out_h, out_w))
    # Perform the convolution
    for i in range(out_h):
        for j in range(out_w):
            # Extract the slice of the image based on the current window
            # location and the stride
            slice_img = (padded_images
                         [:, i * sh: i * sh + kh, j * sw: j * sw + kw])
            # Perform element-wise multiplication and sum it up
            output = np.sum(slice_img * kernel, axis=(1, 2))
            convoluted[:, i, j] = output
    return convoluted
