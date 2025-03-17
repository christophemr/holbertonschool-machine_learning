#!/usr/bin/env python3
"""
Remove rows with NaN values in 'Close' column from a DataFrame.
"""


def prune(df):
    """
    Remove entries where the Close column has NaN values.

    Args:
        df (pd.DataFrame): The input DataFrame containing a Close column.

    Returns:
        pd.DataFrame: The modified DataFrame with NaN values
        in the Close column removed.
    """
    # Remove rows where the Close column has NaN values
    df = df.dropna(subset=['Close'])

    return df
