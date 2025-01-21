#!/usr/bin/env python3
"""
Defines a function that calculates the positional encoding for a transformer
"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates positional encoding for a Transformer.

    Args:
        max_seq_len: Maximum sequence length.
        dm: Model depth.

    Returns:
        A numpy.ndarray of shape (max_seq_len, dm) containing the positional
        encoding vectors.
    """

    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, dm, 2) * -(np.log(10000.0) / dm))
    pos_encoding = np.zeros((max_seq_len, dm))

    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)

    return pos_encoding
