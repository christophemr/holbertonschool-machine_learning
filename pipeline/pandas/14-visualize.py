#!/usr/bin/env python3
"""
Visualize and transform the Dataframe
"""

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file


def visualize(df):
    """
    Transform and visualize the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing a Timestamp column.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    # Remove the Weighted_Price column
    df = df.drop(columns=['Weighted_Price'])

    # Rename the Timestamp column to Date
    df.rename(columns={'Timestamp': 'Date'}, inplace=True)

    # Convert the timestamp values to date values
    df['Date'] = pd.to_datetime(df['Date'], unit='s')

    # Index the DataFrame on Date
    df.set_index('Date', inplace=True)

    # Fill missing values in Close with the previous row's value
    df['Close'] = df['Close'].fillna(method='ffill')

    # Fill missing values in High, Low, and Open with same row's Close value
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # Set missing values in Volume_(BTC) and Volume_(Currency) to 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    # Filter data from 2017 onwards
    df = df[df.index >= '2017']

    # Resample data to daily intervals and aggregate
    daily_df = df.resample('D').agg({
        'High': 'max',
        'Low': 'min',
        'Open': 'mean',
        'Close': 'mean',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum'
    })

    return daily_df


# Load the data
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Transform the data
transformed_df = visualize(df)

# Print the transformed DataFrame
print(transformed_df)

# Plot the data
plt.figure(figsize=(10, 5))
transformed_df.plot(ax=plt.gca())
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Bitcoin Price and Volume Over Time')
plt.legend(loc='best')
plt.show()
