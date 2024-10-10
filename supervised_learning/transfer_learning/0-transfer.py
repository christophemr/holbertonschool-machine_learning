#!/usr/bin/env python3
"""
Script that trains a convolutional neural network to classify the CIFAR-10
dataset using transfer learning with MobileNetV2 (with resized images).
"""

from tensorflow import keras as K


def preprocess_data(X, Y):
    """Preprocess the data for the model.

    Args:
    X: numpy.ndarray of shape (m, 32, 32, 3), CIFAR-10 images
    Y: numpy.ndarray of shape (m,), CIFAR-10 labels

    Returns:
    X_p: Preprocessed images (normalized)
    Y_p: One-hot encoded labels
    """
    X_p = X.astype('float32') / 255.0
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return X_p, Y_p


if __name__ == '__main__':
    # Load CIFAR-10 dataset
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # Preprocessing (normalize and one-hot encode)
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    # Define the input shape for the original CIFAR-10 image size (32x32)
    input_shape = (32, 32, 3)
    base_model_input = K.Input(shape=input_shape)

    # Lambda layer to resize the images from 32x32 to 96x96
    resized_lay = K.layers.Lambda(
        lambda x: K.backend.resize_images(
            x,
            height_factor=(96 // 32),
            width_factor=(96 // 32),
            data_format="channels_last",
            interpolation='bilinear'   # Interpolation method for resizing
            ))(base_model_input)

    # Load MobileNetV2 without the top classification layers
    base_model = K.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=resized_lay  # The input tensor after resizing
    )

    # Freeze all layers except the last 10 for fine-tuning
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Add custom classification layers on top of the base model
    x = base_model.output
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.6)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    # Create the final model
    model = K.Model(inputs=base_model_input, outputs=outputs)

    # Use SGD with momentum
    optimizer = K.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=K.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    # Use early stopping callback
    early_stopping = K.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    lr_scheduler = K.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-5
    )
    # Train the model using the data augmentation
    history = model.fit(
        X_train_p, Y_train_p,
        batch_size=256,
        validation_data=(X_test_p, Y_test_p),
        epochs=30,
        verbose=1,
        callbacks=[early_stopping, lr_scheduler]
    )

    # Save the trained model to a file
    model.save('cifar10.h5')

    # Save the training metrics (loss, accuracy) to a CSV file for analysis
    with open('training_metrics.csv', 'w') as f:
        f.write('epoch,loss,accuracy,val_loss,val_accuracy\n')
        for i in range(len(history.history['loss'])):
            f.write('{},{},{},{},{}\n'.format(
                i+1,
                history.history['loss'][i],
                history.history['accuracy'][i],
                history.history['val_loss'][i],
                history.history['val_accuracy'][i]
            ))
