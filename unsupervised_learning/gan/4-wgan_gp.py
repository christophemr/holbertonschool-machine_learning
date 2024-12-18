#!/usr/bin/env python3
"""
Wasserstein GAN with Gradient Penalty (WGAN-GP).
"""

import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """
    Implements a Wasserstein GAN with Gradient Penalty.

    WGAN-GP improves training stability by adding a gradient penalty
    to the discriminator loss and avoiding weight clipping.

    Attributes:
        generator (tf.keras.Model): The generator network.
        discriminator (tf.keras.Model): The discriminator network.
        latent_generator (function): Function to generate latent vectors.
        real_examples (tf.Tensor): Tensor of real examples for training.
        batch_size (int): Batch size for training.
        disc_iter (int): Number of discriminator updates per generator step.
        learning_rate (float): Learning rate for optimizers.
        lambda_gp (float): Weight of the gradient penalty term.
        axis (tf.Tensor): Axis for computing gradient penalty norms.
        scal_shape (tf.Tensor): Shape for interpolating samples.
    """

    def __init__(self, generator, discriminator,
                 latent_generator, real_examples,
                 batch_size=200, disc_iter=2,
                 learning_rate=0.005, lambda_gp=10):
        """
        Initializes the WGAN-GP model.

        Args:
            generator (tf.keras.Model): Generator network.
            discriminator (tf.keras.Model): Discriminator network.
            latent_generator (function): Function to generate latent vectors.
            real_examples (tf.Tensor): Tensor of real examples.
            batch_size (int): Batch size. Defaults to 200.
            disc_iter (int): Discriminator iterations per step.
            Defaults to 2.
            learning_rate (float): Learning rate for optimizers.
            Defaults to 0.005.
            lambda_gp (float, optional): Gradient penalty weight.
            Defaults to 10.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.3  # Adam optimizer beta_1 parameter
        self.beta_2 = 0.9  # Adam optimizer beta_2 parameter
        self.lambda_gp = lambda_gp

        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')

        # Shape for interpolation
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # Define generator loss and optimizer
        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(optimizer=self.generator.optimizer,
                               loss=self.generator.loss)

        # Define discriminator loss and optimizer
        self.discriminator.loss = (
            lambda x, y: tf.reduce_mean(y) - tf.reduce_mean(x))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(optimizer=self.discriminator.optimizer,
                                   loss=self.discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """
        Generates fake samples using the generator.

        Args:
            size (int, optional): Number of samples to generate. Defaults
            to batch_size.
            training (bool, optional): Training mode flag. Defaults to False.

        Returns:
            tf.Tensor: Generated fake samples.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        Retrieves a batch of real samples from the dataset.

        Args:
            size (int, optional): Number of real samples. Defaults
            to batch_size.

        Returns:
            tf.Tensor: Randomly selected real samples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Generates interpolated samples between real and fake samples.

        Args:
            real_sample (tf.Tensor): Real samples.
            fake_sample (tf.Tensor): Fake samples.

        Returns:
            tf.Tensor: Interpolated samples.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Computes the gradient penalty for the discriminator.

        Args:
            interpolated_sample (tf.Tensor): Interpolated samples.

        Returns:
            tf.Tensor: The gradient penalty value.
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def replace_weights(self, gen_h5, disc_h5):
        """Replaces the weights of the generator and discriminator."""
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)
        print(f"Weights successfully loaded from {gen_h5} and {disc_h5}.")

    def train_step(self, useless_argument):
        """
        Performs one training step for the WGAN-GP.

        This includes:
            - Training the discriminator disc_iter times with gradient penalty
            - Training the generator once.

        Args:
            useless_argument: Unused argument (required by Keras API).

        Returns:
            dict: Dictionary containing losses:
                - "discr_loss": Discriminator loss.
                - "gen_loss": Generator loss.
                - "gp": Gradient penalty.
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)
                interpolated_sample = self.get_interpolated_sample(
                    real_samples, fake_samples)

                # Compute discriminator loss and gradient penalty
                real_preds = self.discriminator(real_samples, training=True)
                fake_preds = self.discriminator(fake_samples, training=True)
                discr_loss = self.discriminator.loss(real_preds, fake_preds)
                gp = self.gradient_penalty(interpolated_sample)
                new_discr_loss = discr_loss + self.lambda_gp * gp

            # Update discriminator
            discr_gradients = tape.gradient(
                new_discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradients, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as tape:
            fake_samples = self.get_fake_sample(training=True)
            fake_preds = self.discriminator(fake_samples, training=False)
            gen_loss = self.generator.loss(fake_preds)

        gen_gradients = tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
