import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.initializers import RandomNormal, Zeros
from tensorflow.keras.layers import (Concatenate, Conv2DTranspose, Dense,
                                     Dropout, Input, LeakyReLU, Permute,
                                     Reshape)

#from src.utils.gan_losses import pretrain_generator_loss

RANDOM_SEED = 42

def make_generator(
    image_dim=[3, 64, 64],
    latent_dim=1280,
    conditions_dim=6,
    len_input_seq=2,
    len_output_seq=2,
    n_filters=[32, 16, 64],
    kernel_size=[5, 3, 1],
    stride=[1, 1, 1],
    padding=['same', 'same', 'same'],
    hidden_dims=[128, 128],
    dropout=0.3,
    alpha=0.2,
    activation='leaky_relu',
    image_mean = 0.36472765,
    image_std = 0.33011988,
    random_seed=RANDOM_SEED
):
    # Input layers
    X_input = Input(shape=(len_input_seq, *image_dim), dtype='float32')
    X_reshaped = Permute((1, 3, 4, 2))(X_input)

    z_input = Input(shape=(len_input_seq, latent_dim,), dtype='float32')
    c_input = Input(shape=(len_input_seq-1, conditions_dim,), dtype='float32')

    cz_input = z_input[:, 0, :]
    for i in range(c_input.shape[1]):
        z = z_input[:, i+1, :]
        c = c_input[:, i, :]
        cz_input = Concatenate()([cz_input, c, z])

    output_channels = image_dim[0]

    x = cz_input

    for dim in hidden_dims:
        x = Dense(
            dim,
            activation=activation,
            kernel_initializer=RandomNormal(stddev=0.01, seed=random_seed),
            bias_initializer=Zeros(),
            )(x)
        x = Dropout(dropout)(x)
        x = LeakyReLU(alpha=alpha)(x)

    x = Dense(
        np.prod(n_filters[0] * 7 * 7),
        activation=activation,
        kernel_initializer=RandomNormal(stddev=0.01, seed=random_seed),
        bias_initializer=Zeros(),
        )(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Reshape((7, 7, n_filters[0]))(x) # dimension of the first layer of the generator, chosen arbitrarily: 7

    last_kernel_size = (image_dim[1] - (x.shape[1] - 1) * 1 + 2 * 0) #(output_dim - (input_dim - 1) * stride + 2 * padding)
    x = Conv2DTranspose(
            filters=output_channels,
            kernel_size=last_kernel_size,
            strides=1,
            padding='valid', # valid maeans no padding
            activation='sigmoid',
            kernel_initializer=RandomNormal(mean=image_mean, stddev=image_std, seed=random_seed),
            bias_initializer=Zeros(),
        )(x)

    # Concatenate the intermediate output with the input images
    previous_layer = layers.Concatenate()([x, X_reshaped[:, 0, :, :, :]])
    """for j in range(1, len_input_seq):
        previous_layer = layers.Concatenate()([previous_layer, X_reshaped[:, j, :, :, :]])"""

    for j in range(len_input_seq+len_output_seq-1):
        for i in range(len(n_filters)):
            x_output = Conv2DTranspose(
                filters=n_filters[i],
                kernel_size=kernel_size[i],
                strides=stride[i],
                padding=padding[i],
                activation=activation,
                kernel_initializer=RandomNormal(mean=image_mean, stddev=image_std, seed=random_seed),
                bias_initializer=Zeros(),
            )(previous_layer)
        
        last_kernel_size = (image_dim[1] - (x_output.shape[1] - 1) * 1 + 2 * 0) #(output_dim - (input_dim - 1) * stride + 2 * padding)
        x_output = Conv2DTranspose(
                filters=output_channels,
                kernel_size=last_kernel_size,
                strides=1,
                padding='valid', # valid maeans no padding
                activation='sigmoid',
                kernel_initializer=RandomNormal(mean=image_mean, stddev=image_std, seed=random_seed),
                bias_initializer=Zeros(),
            )(x_output)
        
        if j < len_input_seq-1:
            x_output = X_reshaped[:, j+1, :, :, :]  # Teacher forcing

        # Concatenate contestual information with the output of the previous layer
        previous_layer = Concatenate()([x, x_output])

        if j >= len_input_seq-1:
            # Reshape the output to match the input image dimensions and concatenate the output layers
            x_output = Permute((3, 1, 2))(x_output) # (channels, height, width)
            x_output = Reshape((1, *image_dim))(x_output)

            output_layers = x_output if j==(len_input_seq-1) else layers.Concatenate(axis=1)([output_layers, x_output])

    model = Model(inputs=[X_input, z_input, c_input], outputs=output_layers)
    model.compile(optimizer='adam', loss='mse')

    return model

# Test the generator model
if __name__ == '__main__':
    image_dim = (3, 64, 64)
    latent_dim = 1280
    conditions_dim = 6
    len_input_seq = 2
    len_output_seq = 2

    model = make_generator(
        image_dim,
        latent_dim,
        conditions_dim,
        len_input_seq,
        len_output_seq,
        n_filters=[4, 4, 4, 4],
        kernel_size=[5, 5, 3, 1],
        stride=[2, 1, 2, 1],
        padding=['same', 'same', 'same', 'same'],
        activation='relu'
    )