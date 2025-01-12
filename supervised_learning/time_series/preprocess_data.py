#!/usr/bin/env python3
"""
Preprocess Bitcoin (BTC) data for training and validation
"""

import pandas as pd
import numpy as np


def preprocess_data(
        input_file, output_file, sliding_window=24, smoothing_window=3):
    """
    Preprocess Bitcoin data for RNN model training.

    Args:
        input_file (str): Path to the raw dataset (CSV file).
        output_file (str): Path to save the processed dataset.
        sliding_window (int): Number of previous hours to consider for
        forecasting.
        smoothing_window (int): Window size for moving average smoothing
    Saves:
        Processed data as a .npz file containing X and y.
    """
    # Load dataset
    df = pd.read_csv(input_file)

    # Convert Timestamp to datetime and set as index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.set_index('Timestamp', inplace=True)

    # Drop rows with all NaN values
    df.dropna(how='all', inplace=True)

    # Resample data into hourly intervals and calculate the mean
    hourly_data = df.resample('1h').mean()

    # Fill missing values
    hourly_data = hourly_data.ffill().bfill()

    # Select relevant features
    features = ["Close", "Volume_(BTC)", "Weighted_Price"]
    hourly_data = hourly_data[features]

    # Apply moving average smoothing (optional)
    if smoothing_window:
        hourly_data["Close"] = (
            hourly_data["Close"].rolling(window=smoothing_window).mean())
        hourly_data = hourly_data.dropna()

    # Normalize features and target
    normalized_data = (hourly_data - hourly_data.mean()) / hourly_data.std()

    # Save normalization parameters for later use
    close_mean = hourly_data["Close"].mean()
    close_std = hourly_data["Close"].std()

    # Prepare sliding windows for training
    X, y = [], []
    for i in range(len(normalized_data) - sliding_window):
        X.append(normalized_data.iloc[i:i + sliding_window].values)
        y.append(
            (hourly_data["Close"].iloc[i + sliding_window] - close_mean) / close_std)

    # Convert to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # Save processed data and normalization parameters
    np.savez(output_file, X=X, y=y, close_mean=close_mean, close_std=close_std)


if __name__ == "__main__":
    input_file = "coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"
    output_file = "processed_data.npz"
    preprocess_data(input_file, output_file)
