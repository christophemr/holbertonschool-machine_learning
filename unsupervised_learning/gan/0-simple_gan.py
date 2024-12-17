#!/usr/bin/env python3
"""
Simple GAN implementation using TensorFlow/Keras
"""

import tensorflow as tf
from tensorflow import keras


class Simple_GAN(keras.Model):
    """
    Simple Generative Adversarial Network (GAN) class.

    Attributes:
        generator (keras.Model): The generator network.
        discriminator (keras.Model): The discriminator network.
        latent_generator (function): A function to generate latent
        space input (random noise).
        real_examples (tf.Tensor): Tensor of real examples for training.
        batch_size (int): Batch size for training. Default is 200.
        disc_iter (int): Number of discriminator updates per generator
        update. Default is 2.
        learning_rate (float): Learning rate for both generator and
        discriminator. Default is 0.005.
        beta_1 (float): Beta_1 parameter for Adam optimizer.
        beta_2 (float): Beta_2 parameter for Adam optimizer.
    """

    def __init__(self, generator, discriminator,
                 latent_generator, real_examples,
                 batch_size=200, disc_iter=2,
                 learning_rate=0.005):
        """
        Initializes the Simple_GAN model.

        Args:
            generator (keras.Model): The generator network.
            discriminator (keras.Model): The discriminator network.
            latent_generator (function):function to generate latent space input
            real_examples (tf.Tensor): Tensor of real examples for training.
            batch_size (int): Batch size for training. Default is 200.
            disc_iter (int): Number of discriminator updates per
            generator update.
            Default is 2.
            learning_rate (float): Learning rate for optimizers.
            Default is 0.005.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        # Optimizer parameters
        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) +
            tf.keras.losses.MeanSquaredError()(y, -1*tf.ones(y.shape)))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """
        Generates fake samples using the generator.

        Args:
            size (int): Number of samples to generate. Defaults to batch_size.
            training (bool): If True, run in training mode. Default is False.

        Returns:
            tf.Tensor: Generated fake samples.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        Retrieves a batch of real samples.

        Args:
            size (int): Number of real samples to retrieve.

        Returns:
            tf.Tensor: Real samples.
        """
        if not size:
            size = self.batch_size
        indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """
        Performs one training step for the GAN, including:
        - Training the discriminator multiple times (self.disc_iter).
        - Training the generator once.

        Args:
            useless_argument: Placeholder argument to match the Keras API.

        Returns:
            dict: Dictionary containing discriminator loss and generator loss.
        """
        # Train the discriminator
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                # Get real and fake samples
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)

                # Compute discriminator loss
                real_preds = self.discriminator(real_samples)
                fake_preds = self.discriminator(fake_samples)
                discr_loss = self.discriminator.loss(real_preds, fake_preds)

            # Update discriminator weights
            grads = tape.gradient(
                discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as tape:
            # Generate fake samples
            fake_samples = self.get_fake_sample(training=True)
            fake_preds = self.discriminator(fake_samples)

            # Compute generator loss
            gen_loss = self.generator.loss(fake_preds)

        # Update generator weights
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables))

        # Return losses for monitoring
        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
