#!/usr/bin/env python3
"""
Slicing specific columns and rows from a DataFrame.
"""

import pandas as pd


def slice(df):
    """
    Extract specific columns and select every 60th row from a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing the necessary columns.

    Returns:
        pd.DataFrame: The sliced DataFrame with selected columns and rows.
    """
    # Extract the columns High, Low, Close, and Volume_(BTC)
    selected_cols = df[['High', 'Low', 'Close', 'Volume_(BTC)']]

    # Select every 60th row
    sliced_df = selected_cols.iloc[::60]

    return sliced_df
