#!/usr/bin/env python3
"""
Clean and fill missing values in a DataFrame.
"""


def fill(df):
    """
    Modify the DataFrame by removing the Weighted_Price column,
    filling missing values,and setting specific missing values to 0.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    # Remove the Weighted_Price column
    df = df.drop(columns=['Weighted_Price'])

    # Fill missing values in the Close column with the previous row's value
    df['Close'] = df['Close'].fillna(method='ffill')

    # Fill missing values in the High, Low, and Open columns with the
    # corresponding Close value
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # Set missing values in Volume_(BTC) and Volume_(Currency) to 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
