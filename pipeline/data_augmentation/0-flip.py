#!/usr/bin/env python3
"""
Function that flips an image horizontally
"""

import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally.

    Args:
        image: A 3D tf.Tensor containing the image to flip.
           The dimensions are expected to be [height, width, channels].

    Returns:
        The horizontally flipped image as a tf.Tensor.
    """
    return tf.image.flip_left_right(image)
