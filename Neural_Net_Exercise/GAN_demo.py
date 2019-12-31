################################################################################
# CS 156a Bonus Exercise Addendum
# Author: George Stathopoulos
# Last modified: October 30, 2019
# Description: A script to train a GAN on MNIST. Generates image samples from
#              the generator and saves to a local directory, does not save the
#              models.
################################################################################

import matplotlib
matplotlib.use('Agg')

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Conv2D, Conv2DTranspose, Flatten
from keras.layers import Reshape, UpSampling2D, MaxPooling2D
from keras.layers import LeakyReLU, Input
from keras.layers import BatchNormalization

from train import get_data
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# Some constants
MNIST_SIZE = 28
LATENT_DIM = 100

def make_generator(num_filters=64, num_hidden_conv_layers=2, init_dim=7):
    gen = Sequential()
    # Model input is a feature vector of size 100
    gen.add(Dense(init_dim**2 * num_filters, input_dim=LATENT_DIM))
    gen.add(Activation('relu'))
    gen.add(Reshape((init_dim, init_dim, num_filters)))

    for _ in range(num_hidden_conv_layers):
        # Input: d x d x k
        # Output 2d x 2d x k/2
        if (init_dim < MNIST_SIZE):
            gen.add(UpSampling2D())
            init_dim *= 2
        num_filters //= 2
        gen.add(Conv2DTranspose(num_filters, 5, padding='same'))
        gen.add(BatchNormalization(momentum=0.4))
        gen.add(Activation('relu'))

    gen.add(Conv2DTranspose(1, 5, padding='same'))
    gen.add(Activation('sigmoid'))
    # Output should be 28 x 28 x 1
    # gen.summary()
    return gen

def make_discriminator(num_filters=32, num_hidden_layers=3, dropout=0.3):
    d = Sequential()

    d.add(Conv2D(num_filters*1, 5, strides=2,
                 input_shape=(MNIST_SIZE, MNIST_SIZE, 1), padding='same'))
    d.add(LeakyReLU()) # leakyrelu so generator has derivative
    d.add(Dropout(dropout))

    for i in range(1, num_hidden_layers):
        # Powers of 2 are generally better suited for GPU
        d.add(Conv2D(num_filters*(2**i), 5, strides=2, padding='same'))
        d.add(LeakyReLU())
        d.add(Dropout(dropout))

    # NOTE: Difference between this and build_conv_net
    #       is that there is only a SINGLE output class,
    #       which corresponds to FAKE/REAL.
    d.add(Flatten())
    d.add(Dense(1))
    d.add(Activation('sigmoid'))
    d.compile(loss='binary_crossentropy', optimizer='adam')
    return d

def make_adversial_network(generator, discriminator):
    # This will only be used for training the generator.
    # Note, the weights in the discriminator and generator are shared.
    discriminator.trainable = False
    gan = Sequential([generator, discriminator])
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan #, generator, discriminator

def generate_latent_noise(n):
    return np.random.uniform(-1, 1, size=(n, LATENT_DIM))

def visualize_generator(epoch, generator,
                        num_samples=100, dim=(10,10),
                        figsize=(10,10), path=''):
    plt.figure(figsize=figsize)
    for i in range(num_samples):
        plt.subplot(dim[0], dim[1], i+1)
        img = generator.predict(generate_latent_noise(1))[0,:,:,0]
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'generator_samples/gan_epoch_{epoch}.png')
    plt.close()

def train(epochs=1, batch_size=128, path=''):
    # Import the MNIST dataset using Keras, will only
    # use the 60,000 training examples.
    (X_train, _), _ = get_data(True)

    # Creating GAN
    generator     = make_generator()
    discriminator = make_discriminator()
    adversial_net = make_adversial_network(generator, discriminator)

    visualize_generator(0, generator, path=path)
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}')

        discr_loss = 0
        gen_loss = 0
        for _ in tqdm(range(batch_size)):
            noise = generate_latent_noise(batch_size)
            generated_images = generator.predict(noise)

            real_images = X_train[np.random.choice(X_train.shape[0], batch_size,
                                                   replace=False)]

            discrimination_data = np.concatenate([real_images, generated_images])

            # Labels for generated and real data, uses soft label trick
            discrimination_labels = 0.1 * np.ones(2 * batch_size)
            discrimination_labels[:batch_size] = 0.9

            # To train, we alternate between training just the discriminator
            # and just the generator.
            discriminator.trainable = True
            discr_loss += discriminator.train_on_batch(discrimination_data,
                                                       discrimination_labels)

            # Trick to 'freeze' discriminator weights in adversial_net. Only
            # the generator weights will be changed, which are shared with
            # the generator.
            discriminator.trainable = False
            # N.B, changing the labels because now we want to 'fool' the
            # discriminator.
            gen_loss += adversial_net.train_on_batch(noise, np.ones(batch_size))

        print(f'Discriminator Loss: {discr_loss/batch_size}')
        print(f'Generator Loss:     {gen_loss/batch_size}')
        visualize_generator(epoch+1, generator, path=path)

if __name__ == '__main__':
    # Create img directory to save generator image samples to
    os.makedirs(os.path.join(os.getcwd(), 'generator_samples'), exist_ok=True)
    train(epochs=400)
