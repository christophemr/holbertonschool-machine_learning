#!/usr/bin/env python3
"""
Rearrange MultiIndex and concatenate bitstamp and coinbase tables
for a specific time range.
"""

import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Rearrange the MultiIndex and concatenate specific rows from two DataFrames.

    Args:
        df1 (pd.DataFrame): The first DataFrame (coinbase).
        df2 (pd.DataFrame): The second DataFrame (bitstamp).

    Returns:
        pd.DataFrame: The concatenated DataFrame with labeled rows.
    """
    # Index both dataframes on their Timestamp columns
    df1 = index(df1)
    df2 = index(df2)

    # Filter rows with timestamps between 1417411980 and 1417417980
    df1_filtered = df1[(df1.index >= 1417411980) & (df1.index <= 1417417980)]
    df2_filtered = df2[(df2.index >= 1417411980) & (df2.index <= 1417417980)]

    # Concatenate the filtered rows from both dataframes with keys
    concatenated_df = pd.concat(
        {'coinbase': df1_filtered, 'bitstamp': df2_filtered})

    # Rearrange the MultiIndex so that Timestamp is the first level
    concatenated_df = concatenated_df.swaplevel().sort_index()

    return concatenated_df
