# Transfer Learning - CIFAR-10 Classification Project

## Project Overview

This project explores the application of **Transfer Learning** to train a Convolutional Neural Network (CNN) on the CIFAR-10 dataset. The primary focus is on leveraging pre-trained models from Keras and fine-tuning them to achieve a high accuracy on the CIFAR-10 classification task.

The goal is to obtain a **validation accuracy of 87% or higher** by utilizing transfer learning techniques.

## Learning Objectives

By the end of this project, you should be able to:

- Explain the concept of **transfer learning** and how it can be applied to new tasks.
- Define and understand **fine-tuning** and **frozen layers**.
- Know how and why to freeze layers in a model.
- Apply transfer learning using **Keras applications** in real-world tasks.
  
## Project Tasks

### Task 0: Transfer Knowledge

The first task is to create a Python script that trains a convolutional neural network to classify the CIFAR-10 dataset using **Keras Applications** and transfer learning.

#### Requirements

- Use a pre-trained model from [Keras Applications](https://keras.io/api/applications/).
- Save the trained model as `cifar10.h5` in the current working directory.
- Ensure the saved model is compiled and achieves a **validation accuracy of at least 87%**.
- The script must not run when the file is imported; it should only run when executed directly.

#### Key Hints

1. **Training Time**: The process of training and hyperparameter tuning may take a while, so start early.
2. **Image Size**: The CIFAR-10 dataset contains 32x32 pixel images, while most pre-trained Keras applications are designed for larger images. You will need to resize the images using a Lambda layer in your model.
3. **Freezing Layers**: Freeze most layers of the pre-trained model, and compute the output of these layers only once. This optimization saves significant training time.

#### Function: `preprocess_data`

You will also implement a function called `preprocess_data` to pre-process the CIFAR-10 dataset before passing it through the model.

```python
def preprocess_data(X, Y):
    """
    Pre-processes the data for the model.

    Parameters:
    - X (numpy.ndarray): Shape (m, 32, 32, 3), the CIFAR-10 images.
    - Y (numpy.ndarray): Shape (m,), the CIFAR-10 labels for X.

    Returns:
    - X_p (numpy.ndarray): Preprocessed images.
    - Y_p (numpy.ndarray): Preprocessed labels.
    """
