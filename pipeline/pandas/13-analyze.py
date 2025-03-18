#!/usr/bin/env python3
"""
Compute descriptive statistics for all columns except Timestamp.
"""


def analyze(df):
    """
    Compute descriptive statistics for all columns except the Timestamp column.

    Args:
        df (pd.DataFrame): The input DataFrame containing a Timestamp column.

    Returns:
        pd.DataFrame: A new DataFrame containing the descriptive statistics.
    """
    # Drop the Timestamp column and compute descriptive statistics
    stats = df.drop(columns=['Timestamp']).describe()

    return stats
