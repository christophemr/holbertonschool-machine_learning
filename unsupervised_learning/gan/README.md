# GAN

## Description

This project explores the basics of **Generative Adversarial Networks (GANs)** through multiple tasks, implementing and improving different GAN architectures to generate synthetic data effectively.

### Tasks

1. **Simple GAN**
   Implement the `Simple_GAN` class, focusing on the `train_step` method to train a basic GAN.

2. **Wasserstein GAN (WGAN)**
   Modify the generator and discriminator loss functions, and clip the discriminator weights to train a WGAN.

3. **Wasserstein GAN with Gradient Penalty (WGAN-GP)**
   Implement a gradient penalty for the discriminator to overcome the limitations of weight clipping in WGAN.

4. **Convolutional GAN for Image Data**
   Develop convolutional generator and discriminator networks to work with image-based datasets, such as face generation.

5. **Face Generator**
   Train a WGAN-GP model to generate realistic faces using a provided dataset of grayscale 16x16 images.

6. **Pre-trained GAN Exploration**
   Load pre-trained model weights to explore and experiment with a GAN that generates high-quality synthetic faces.

---

## Resources

- **Papers**:
   - Goodfellow et al. (2014): Introduction to GANs
   - Arjovsky et al. (2017): Wasserstein GAN
   - Gulrajani et al. (2017): WGAN with Gradient Penalty

- **Concepts**:
   - [Introduction to the Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric)
   - [This Person Does Not Exist](https://thispersondoesnotexist.com)

---

## Skills Required

- Understanding of basic neural networks and Keras/TensorFlow.
- Knowledge of **GANs**, including:
   - Generator and Discriminator architectures
   - Loss functions for adversarial training
   - Wasserstein distance and gradient penalty
- Experience with convolutional neural networks (CNNs).
- Python programming and data manipulation with NumPy.

---

### Project Highlights

- Learn to build and train GAN architectures from scratch.
- Understand the challenges in GAN training, such as vanishing gradients.
- Work with real-world image generation tasks, such as generating realistic faces.

---

## Getting Started

To test and run the project, ensure you have:

- **Python 3.9** or later
- **NumPy 1.25.2**
- **TensorFlow 2.15.0**

---