#!/usr/bin/env python3
"""
Concatenate two DataFrames with timestamp indexing and
multi-key labeling.
"""

import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenate two DataFrames after indexing them on their Timestamp columns.

    Args:
        df1 (pd.DataFrame): The first DataFrame (coinbase).
        df2 (pd.DataFrame): The second DataFrame (bitstamp).

    Returns:
        pd.DataFrame: The concatenated DataFrame with labeled rows.
    """
    # Index both dataframes on their Timestamp columns
    df1 = index(df1)
    df2 = index(df2)

    # Select rows from df2 up to and including timestamp 1417411920
    df2_selected = df2[df2.index <= 1417411920]

    # Concatenate the selected rows from df2 to the top of df1 with keys
    concatenated_df = pd.concat({'bitstamp': df2_selected, 'coinbase': df1})

    return concatenated_df
