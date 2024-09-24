#!/usr/bin/env python3
"""
Defines a function that performs convolution
on a image with multiple channels and multiple kernels
using given padding and stride
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Parameters:
    images (numpy.ndarray): shape (m, h, w, c) with multiple images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the images
    kernels (numpy.ndarray): shape (kh, kw, c, nc) with the kernels
        kh is the height of a kernel
        kw is the width of a kernel
        nc is the number of kernels
    padding (str or tuple): 'same', 'valid', or tuple of (ph, pw)
        ph is the padding for the height
        pw is the padding for the width
    stride (tuple): (sh, sw)
        sh is the stride for the height
        sw is the stride for the width

    Returns:
    numpy.ndarray: containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    # Padding calculation for 'same'
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding

    # Apply padding to the images
    padded_images = np.pad(
      images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    # Output dimensions
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    # Initialize the output array
    output = np.zeros((m, out_h, out_w, nc))

    # Perform convolution
    for i in range(out_h):
        for j in range(out_w):
            for k in range(nc):
                v_start = i * sh
                vert_end = v_start + kh
                horiz_start = j * sw
                horiz_end = horiz_start + kw

                slice_img = (
                  padded_images[:, v_start:vert_end, horiz_start:horiz_end, :])
                output[:, i, j, k] = np.sum(
                  slice_img * kernels[:, :, :, k], axis=(1, 2, 3))
    return output
