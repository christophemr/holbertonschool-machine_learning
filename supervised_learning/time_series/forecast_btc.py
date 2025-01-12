#!/usr/bin/env python3
"""
Train and validate an RNN model for Bitcoin price forecasting
"""
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def load_data(file):
    """
    Load preprocessed data.
    Args:
        file (str): Path to the preprocessed dataset (Numpy file).
    Returns:
        tf.data.Dataset: Datasets for training, validation, and testing.
    """
    file = "processed_data.npz"
    data = np.load(file)
    X, y = data["X"], data["y"]

    # Split data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    # Convert to tf.data.Dataset
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((
            X_train, y_train)).batch(64).shuffle(1000))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)
    test_dataset = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64))

    return train_dataset, val_dataset, test_dataset, X_test, y_test


def build_model(input_shape):
    """
    Build the RNN model.
    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).
    Returns:
        tf.keras.Model: Compiled RNN model.
    """
    model = Sequential([
        LSTM(
            56, activation="tanh",
            return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32, activation="tanh"),
        Dropout(0.3),
        Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


if __name__ == "__main__":
    data_file = "processed_data.npz"
    (train_dataset, val_dataset,
     test_dataset, X_test, y_test) = load_data(data_file)

    # Build the model
    input_shape = train_dataset.element_spec[0].shape[1:]
    model = build_model(input_shape)

    # Train the model and capture history
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        patience=3,
        factor=0.5,
        verbose=1)
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True, verbose=1)
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20, callbacks=[early_stopping, lr_scheduler, checkpoint])

    test_loss, test_mae = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    # Save the model
    model.save("btc_forecast_model.h5")

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Generate predictions on the test set
    y_test_pred = model.predict(X_test)

    # Plot true vs predicted values (Zoomed view for better visualization)
    sample_indices = 200
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:sample_indices], label='True Values', alpha=0.7)
    plt.plot(
        y_test_pred.flatten()[:sample_indices],
        label='Predictions', alpha=0.7)
    plt.title('True vs Predicted Values (Zoomed Test Set)')
    plt.xlabel('Time')
    plt.ylabel('BTC Price (Normalized)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Scatter plot: True vs Predicted values
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_test_pred.flatten(), alpha=0.5)
    plt.plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)], 'r--', label='Ideal Prediction')
    plt.title('True vs Predicted Values (Scatter Plot)')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()
