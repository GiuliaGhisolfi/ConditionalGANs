import numpy as np
import tensorflow as tf

from src.utils.optical_flow import get_optical_flow_batch

############# GAN LOSS FUNCTIONS #############
"""
    This module contains the loss functions used in the GAN training process.
    The loss functions are used to train the generator and the critic in the WGAN-GP model.

    The loss functions are defined as follows:
        - The critic loss function tries to maximize the critic's output on real data
            and minimize the critic's output on fake data,
            while enforcing the Lipschitz constraint on the critic.

        - The generator loss function tries to minimize the critic's output on fake data,
            the mean squared error between the real and fake images, and the difference between the
            optical flow of the real and fake images.
"""

############# CRITIC LOSS FUNCTIONS #############

def gradient_penalty(critic, real_images, fake_images):
    """
    Compute the gradient penalty for the WGAN-GP loss function
    that enforces the Lipschitz constraint on the critic,
    by penalizing the norm of the gradient of the critic's output

    Args:
        critic: discriminator model
        real_images: real images from the dataset
        fake_images: generated images from the generator

    Returns:
        gradient_penalty: loss value
    """
    epsilon = tf.random.uniform(real_images.shape, 0.0, 1.0)

    mean_real_images = tf.reduce_mean(real_images, axis=0)
    mean_fake_images = tf.reduce_mean(fake_images, axis=0)

    interpolated_images = epsilon * mean_real_images + (1 - epsilon) * mean_fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        interpolated_output = critic(interpolated_images)
    
    gradients = tape.gradient(interpolated_output, interpolated_images)
    gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((gradient_norm - 1.0) ** 2)

    return gradient_penalty

def discriminator_loss(real_output, fake_output, real_images, fake_images, discriminator, gp_weight=0.01):
    """
    Compute the loss for the discriminator in the WGAN-GP loss function
    that tries to maximize the critic's output on real data and minimize the critic's output on fake data,
    while enforcing the Lipschitz constraint on the critic

    Args:
        real_output: output of the discriminator on real images
        fake_output: output of the discriminator on fake images
        real_images: real images from the dataset
        fake_images: generated images from the generator
        discriminator: discriminator model
        gp_weight: weight of the gradient penalty term

    Returns:
        loss value
    """
    # Calculate the loss for the discriminator
    real_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(fake_output), fake_output)
    loss = real_loss + fake_loss

    #loss = -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)
    gp_loss = gp_weight * gradient_penalty(discriminator, real_images, fake_images)

    return loss + gp_weight * gp_loss


############# ACTOR LOSS FUNCTIONS #############

def mean_squared_error(real_images, fake_images):
    """
    Compute the mean squared error between the real and fake images

    Args:
        real_images: real images from the dataset
        fake_images: generated images from the generator

    Returns:
        loss value
    """
    mean_real_images = tf.reduce_mean(real_images, axis=0)
    mean_fake_images = tf.reduce_mean(fake_images, axis=0)

    return tf.reduce_mean(tf.square(mean_real_images - mean_fake_images))

def temporal_consistency_loss(predicted_frame, previous_frame):
    """
    Compute the temporal consistency loss between each frame and the previous frame,
    to ensure that the generated frames are temporally consistent and the motion is smooth

    Args:
        predicted_frame: frame predicted by the generator
        previous_frame: frame from the dataset

    Returns:
        loss value
    """
    target = 0
    for i in range(len(previous_frame)-1):
        target += tf.reduce_mean(tf.abs(previous_frame[i] - previous_frame[i+1]))
    target /= len(previous_frame)-1

    loss = tf.reduce_mean(tf.abs(previous_frame[-1] - predicted_frame[0]))
    for i in range(predicted_frame-1):
        loss += tf.reduce_mean(tf.abs(predicted_frame[i] - predicted_frame[i+1]))
    loss /= len(predicted_frame)-1
    print("loss: ", loss)

    return loss - target

def color_loss(real_images, fake_images):
    """
    Compute the color loss between the real and fake images,
    to ensure that the colors in the generated images are similar to the colors in the real images

    Args:
        real_images: real images from the dataset
        fake_images: generated images from the generator

    Returns:
        loss value
    """
    mean_real_images = tf.reduce_mean(real_images, axis=0)
    mean_fake_images = tf.reduce_mean(fake_images, axis=0)

    loss_r = tf.reduce_mean(tf.abs(mean_real_images[:, 0, :, :] - mean_fake_images[:, 0, :, :]))
    loss_g = tf.reduce_mean(tf.abs(mean_real_images[:, 1, :, :] - mean_fake_images[:, 1, :, :]))
    loss_b = tf.reduce_mean(tf.abs(mean_real_images[:, 2, :, :] - mean_fake_images[:, 2, :, :]))

    return loss_r + loss_g + loss_b

def optical_flow_loss(real_images, fake_images):
    """
    Compute the optical flow loss between the real and fake images,
    to ensure that the motion in the generated images is similar to the motion in the real images

    Args:
        real_images: real images from the dataset
        fake_images: generated images from the generator

    Returns:
        loss value
    """
    optical_flow_target = 0
    optical_flow_predicted = 0

    for images in real_images:
        optical_flow_target += get_optical_flow_batch(images)
    optical_flow_target /= len(real_images)
    
    for images in fake_images:
        optical_flow_predicted += get_optical_flow_batch(images)
    optical_flow_predicted /= len(fake_images)
    
    return tf.reduce_mean(tf.abs(optical_flow_target, optical_flow_predicted))

def generator_loss(real_images, fake_images, fake_output,
    wasserstein_weight=1, flow_weight=1, mse_weight=1, color_weight=1):
    """
    Compute the loss for the generator in the WGAN-GP loss function,
    which tries to minimize the critic's output on fake data,
    the mean squared error between the real and fake images,
    and the difference between the optical flow of the real and fake images

    Args:
        real_images: real images from the dataset
        fake_images: generated images from the generator
        fake_output: output of the critic on fake images
        wasserstein_weight: weight of the Wasserstein loss term
        flow_weight: weight of the optical flow loss term
        mse_weight: weight of the mean squared error loss term

    Returns:
        loss: loss value
        wasserstein_loss: Wasserstein loss value
        optical_flow_loss: optical flow loss value
        mse: mean squared error loss value
    """
    # Compute the Wasserstein loss for the generator
    wasserstein_loss = -tf.reduce_mean(fake_output)

    # Compute the optical flow loss between the real and fake images
    optical_flow_loss_value = optical_flow_loss(real_images, fake_images)
    if optical_flow_loss_value is np.nan:
        optical_flow_loss_value = 2

    # Compute the mean squared error between the real and fake images
    mse = mean_squared_error(real_images, fake_images)

    # Compute color loss
    color_loss_value = color_loss(real_images, fake_images)

    # Combine the losses with the specified weights
    loss = (wasserstein_weight * wasserstein_loss + flow_weight * optical_flow_loss_value +
        mse_weight * mse + color_weight * color_loss_value)

    return loss, wasserstein_loss, optical_flow_loss_value, mse, color_loss_value