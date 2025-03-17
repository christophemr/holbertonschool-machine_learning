#!/usr/bin/env python3
"""
Renames the Timestamp column to Datetime, converts it to datetime format,
and returns a DataFrame with only Datetime and Close columns.
"""

import pandas as pd


def rename(df):
    """
    Rename the Timestamp column to Datetime and convert its values to datetime.
    Display only the Datetime and Close columns.

    Args:
        df (pd.DataFrame): The input DataFrame containing a Timestamp column.

    Returns:
        pd.DataFrame: The modified DataFrame with Datetime and Close columns.
    """
    # Rename the Timestamp column to Datetime
    df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)

    # Convert the Datetime column to datetime objects
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # Select only the Datetime and Close columns
    df = df[['Datetime', 'Close']]

    return df
