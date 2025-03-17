#!/usr/bin/env python3
"""
Selects last 10 rows of High and Close columns from a DataFrame and
converts them to a numpy.ndarray.
"""


def array(df):
    """
    Select the last 10 rows of the High and Close columns from a DataFrame
    and convert them into a numpy.ndarray.

    Args:
        df (pd.DataFrame): Input DataFrame containing High and Close columns.

    Returns:
        numpy.ndarray: The selected values as a numpy array.
    """
    # Select the last 10 rows of the High and Close columns
    selected_rows = df[['High', 'Close']].tail(10)

    # Convert the selected rows into a numpy array
    result_array = selected_rows.to_numpy()

    return result_array
