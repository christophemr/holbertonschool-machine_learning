#!/usr/bin/env python3
"""
Sort a DataFrame by High price in descending order.
"""


def high(df):
    """
    Sort the DataFrame by the High price in descending order.

    Args:
        df (pd.DataFrame): Input DataFrame containing a High column.

    Returns:
        pd.DataFrame: The sorted DataFrame.
    """
    # Sort the DataFrame by the High column in descending order
    sorted_df = df.sort_values(by='High', ascending=False)

    return sorted_df
