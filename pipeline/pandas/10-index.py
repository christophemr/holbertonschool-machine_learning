#!/usr/bin/env python3
"""
Set Timestamp column as index of DataFrame.
"""


def index(df):
    """
    Set the Timestamp column as the index of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing a Timestamp column.

    Returns:
        pd.DataFrame: The modified DataFrame with Timestamp as the index.
    """
    # Set the Timestamp column as the index
    df = df.set_index('Timestamp')

    return df
