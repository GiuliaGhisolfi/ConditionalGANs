import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal, Zeros
from tensorflow.keras.layers import (Conv2D, Conv3D, Dense, Dropout, Flatten,
                                     Input, LeakyReLU, Permute)
from tensorflow.keras.models import Model

RANDOM_SEED = 42

############# DISCRIMINATOR #############
"""
    This module contains the discriminator model used in the GAN training process.
    The discriminator model is used to classify the input images as real or fake.

    The discriminator model is defined as follows:
        - The input layer takes the input images and passes them through a series of convolutional layers
            to extract features from the images, followed by a series of fully connected layers.
        - The output layer is a single unit that outputs the probability of the input image being real or fake,
            using a sigmoid activation function to output a value between 0 and 1.
"""

def make_discriminator(
    input_dim,
    len_input_seq=1,
    n_filters=[4, 4],
    kernel_size=[3, 3],
    stride=[1, 2],
    padding=['same', 'same'],
    hidden_dims=[512, 256, 128, 64],
    activation='relu',
    dropout=0.4,
    alpha=0.2,
    random_seed=RANDOM_SEED,
):
    if len_input_seq == 1:
        input_layer = Input(shape=input_dim, dtype='float32')

        x_image = Permute((2, 3, 1))(input_layer) # from (channels, height, width) to (height, width, channels)
        for i in range(len(n_filters)):
                x_image = Conv2D(
                    filters=n_filters[i],
                    kernel_size=kernel_size[i],
                    strides=stride[i],
                    padding=padding[i],
                    activation=activation,
                    kernel_initializer=RandomNormal(stddev=0.01, seed=random_seed),
                    bias_initializer=Zeros(),
                )(x_image)
        x = Flatten()(x_image)

        #x = Concatenate()([x_image, condition_layer])

        for dim in hidden_dims:
            x = Dense(
                dim,
                activation=activation,
                kernel_initializer=RandomNormal(stddev=0.01, seed=random_seed),
                bias_initializer=Zeros(),
                )(x)
            x = Dropout(dropout)(x)
            x = LeakyReLU(alpha=alpha)(x)
        
        # Image classification layer (binary classification: real or fake)
        image_classification = Dense(
            1,  # Single output unit for binary classification
            activation='sigmoid',  # Sigmoid activation for probability output
            kernel_initializer=RandomNormal(stddev=0.01, seed=random_seed),
            bias_initializer=Zeros(),
        )(x)

        return Model(inputs=input_layer, outputs=image_classification, name='discriminator')

    else:
        input_layer = Input(shape=(len_input_seq, *input_dim), dtype='float32')

        x = input_layer
        for i in range(len(n_filters)):
            x = Conv3D(
                filters=n_filters[i],
                kernel_size=kernel_size[i],
                strides=stride[i],
                padding=padding[i],
                activation=activation,
                kernel_initializer=RandomNormal(stddev=0.01, seed=random_seed),
                bias_initializer=Zeros(),
            )(x)
        x = Flatten()(x)

        for dim in hidden_dims:
            x = Dense(
                dim,
                activation=activation,
                kernel_initializer=RandomNormal(stddev=0.01, seed=random_seed),
                bias_initializer=Zeros(),
                )(x)
            x = Dropout(dropout)(x)
            x = LeakyReLU(alpha=alpha)(x)

        # Image classification layer (binary classification: real or fake)
        image_classification = Dense(
            1,  # Single output unit for binary classification
            activation='sigmoid',  # Sigmoid activation for probability output
            kernel_initializer=RandomNormal(stddev=0.01, seed=random_seed),
            bias_initializer=Zeros(),
        )(x)

        model = Model(inputs=input_layer, outputs=image_classification, name='discriminator')
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

