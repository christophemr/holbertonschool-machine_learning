#!/usr/bin/env python3
"""
Flip and transpose a DataFrame.
"""


def flip_switch(df):
    """
    Sort the DataFrame in reverse chronological order and transpose it.

    Args:
        df (pd.DataFrame): The input DataFrame containing a Timestamp column.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    # Sort the DataFrame by the Timestamp column in descending order
    sorted_df = df.sort_values(by='Timestamp', ascending=False)

    # Transpose the sorted DataFrame
    transposed_df = sorted_df.transpose()

    return transposed_df
