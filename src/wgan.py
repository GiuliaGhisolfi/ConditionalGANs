import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model

from src.utils.gan_losses import discriminator_loss, generator_loss


class WassersteinGAN:
    def __init__(
            self,
            generator,
            discriminator,
            len_input_seq,
            len_generated_seq,
            patience=10,
            save_path=None,
            learning_rate_generator=0.0001,
            learning_rate_discriminator=0.0001,
            n_critic=5,  # number of critic updates per generator update
            clip_value=0.01,  # clipping value for the critic's weights
            discriminator_gradient_penalty_weight=0.01,
            generator_wloss_weight=1.0,
            generator_flow_loss_weight=0.01,
            generator_mse_weight=0.01,
            color_weight=0.01
        ):
        self.generator = generator
        self.discriminator = discriminator

        self.patience = patience
        self.learning_rate_generator = learning_rate_generator
        self.learning_rate_discriminator = learning_rate_discriminator
        self.n_critic = n_critic
        self.clip_value = clip_value
        self.gp_weight = discriminator_gradient_penalty_weight
        self.loss_weight = generator_wloss_weight
        self.flow_weight = generator_flow_loss_weight
        self.mse_weight = generator_mse_weight
        self.color_weight = color_weight

        self.len_input_seq = len_input_seq
        self.len_generated_seq = len_generated_seq

        # Boolean flags for reshaping sequences in order to give them to the discriminator
        self.reshape_generated_seq_check = self.len_input_seq < self.len_generated_seq
        self.reshape_real_seq_check = self.len_input_seq > self.len_generated_seq

        self.save_path = save_path if save_path is not None else 'wgan.h5'

    def clip_weights(self):
        for layer in self.discriminator.layers:
            weights = layer.get_weights()
            weights = [tf.clip_by_value(w, -self.clip_value, self.clip_value) for w in weights]
            layer.set_weights(weights)

    def reshape_seq(self, sequences, len_seq):
        # cut generated sequence to match the input sequence length
        reshaped_generated_sequences = sequences[:, :len_seq]

        for i in range(1, self.len_generated_seq - len_seq + 1):
            reshaped_generated_sequences = tf.concat([
                reshaped_generated_sequences,
                sequences[:, i:i+len_seq]
                ], axis=0)
        return reshaped_generated_sequences

    def train(self, z_train, X_train, cX_train, z_val, X_val, cX_val, epochs, batch_size):
        # Initialize optimizers
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_generator, beta_1=0.5, beta_2=0.9)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_discriminator, beta_1=0.5, beta_2=0.9)

        # Arrays to store training and validation losses
        self.train_losses_gen = []
        self.val_losses_gen = []
        self.train_losses_disc = []
        self.val_losses_disc = []

        # Arrays to store Wasserstein, flow and MSE losses (components of the generator loss)
        self.wasserstein_losses = []
        self.flow_losses = []
        self.mse_losses = []
        self.color_losses = []
        self.val_wasserstein_losses = []
        self.val_flow_losses = []
        self.val_mse_losses = []
        self.val_color_losses = []

        # Variables for early stopping
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        # Break the training loop if the generator loss is None
        break_train = False

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')

            # Shuffle training data at the beginning of each epoch
            indices = tf.random.shuffle(tf.range(start=0, limit=tf.shape(z_train)[0], dtype=tf.int32))
            z_train_shuffled = tf.gather(z_train, indices)
            X_train_shuffled = tf.gather(X_train, indices)
            cX_train_shuffled = tf.gather(cX_train, indices)

            num_batches = X_train.shape[0] // batch_size

            for batch in range(num_batches):
                # Get batch data
                z_batch = z_train_shuffled[batch * batch_size:(batch + 1) * batch_size]
                X_batch = X_train_shuffled[batch * batch_size:(batch + 1) * batch_size]
                cX_batch = cX_train_shuffled[batch * batch_size:(batch + 1) * batch_size]

                if self.reshape_real_seq_check:
                    X_batch_discr = self.reshape_seq(X_batch, len_seq=self.len_generated_seq)
                else:
                    X_batch_discr = X_batch

                for _ in range(self.n_critic):
                    with tf.GradientTape() as disc_tape:
                        # Generate fake images
                        generated_images = self.generator([X_batch, z_batch, cX_batch], training=True)
                        if self.reshape_generated_seq_check:
                            generated_images = self.reshape_seq(generated_images, len_seq=self.len_input_seq)

                        # Get the discriminator's output for real and fake images
                        real_output = self.discriminator(X_batch_discr, training=True)
                        fake_output = self.discriminator(generated_images, training=True)

                        # Calculate the discriminator's loss (Wasserstein loss)
                        disc_loss = discriminator_loss(real_output, fake_output,
                            X_batch_discr, generated_images, self.discriminator, gp_weight=self.gp_weight)

                    # Calculate and apply gradients
                    gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
                    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

                    # Clip discriminator weights
                    self.clip_weights()

                # Train the generator after updating the discriminator
                with tf.GradientTape() as gen_tape:
                    generated_images = self.generator([X_batch, z_batch, cX_batch], training=True)
                    if self.reshape_generated_seq_check:
                        generated_images = self.reshape_seq(generated_images, len_seq=self.len_input_seq)

                    fake_output = self.discriminator(generated_images, training=True)

                    X_batch = self.reshape_seq(X_batch, len_seq=self.len_generated_seq)
                    
                    # Calculate the generator's loss (Wasserstein loss)
                    gen_loss, w_loss, optical_flow_loss, mse, color_loss = generator_loss(X_batch, generated_images,
                        fake_output, wasserstein_weight=self.loss_weight, flow_weight=self.flow_weight, 
                        mse_weight=self.mse_weight, color_weight=self.color_weight)
                    
                if tf.math.is_nan(gen_loss):
                    break_train = True
                    break

                # Calculate and apply gradients
                gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

                print(f'Batch {batch + 1}/{num_batches} - Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}')
            
            if break_train:
                print('Training stopped due to None loss value')
                break # break the training loop if the generator loss is None
            
            # Evaluate the model on the validation set
            generated_images_val = self.generator([X_val, z_val, cX_val], training=False)
            if self.reshape_generated_seq_check:
                generated_images_val = self.reshape_seq(generated_images_val, len_seq=self.len_input_seq)
            if self.reshape_real_seq_check:
                X_val_discr = self.reshape_seq(X_val, len_seq=self.len_generated_seq)
            else:
                X_val_discr = X_val

            real_output_val = self.discriminator(X_val_discr, training=False)
            fake_output_val = self.discriminator(generated_images_val, training=False)

            val_gen_loss, val_w_loss, val_optical_flow_loss, val_mse, val_color_loss = generator_loss(
                X_val_discr, generated_images_val,
                fake_output_val, wasserstein_weight=self.loss_weight, flow_weight=self.flow_weight,
                mse_weight=self.mse_weight, color_weight=self.color_weight)
            
            val_disc_loss = discriminator_loss(real_output_val, fake_output_val,
                X_val_discr, generated_images_val, self.discriminator, gp_weight=self.gp_weight)

            print(f'Generator - Train: {gen_loss}, Val: {val_gen_loss}')
            print(f'Discriminator - Train: {disc_loss}, Val: {val_disc_loss}')

            # Store losses
            self.train_losses_gen.append(gen_loss)
            self.val_losses_gen.append(val_gen_loss)
            self.train_losses_disc.append(disc_loss)
            self.val_losses_disc.append(val_disc_loss)

            self.wasserstein_losses.append(w_loss)
            self.flow_losses.append(optical_flow_loss)
            self.mse_losses.append(mse)
            self.color_losses.append(color_loss)

            self.val_wasserstein_losses.append(val_w_loss)
            self.val_flow_losses.append(val_optical_flow_loss)
            self.val_mse_losses.append(val_mse)
            self.val_color_losses.append(val_color_loss)

            # Save images generated by the model
            for i in range(0, len(X_val), int(len(X_val) / 4)): # Save 5 images
                self.save_generated_image(X_val[i], z_val[i], cX_val[i],
                path=f'generated_images/wgan_train/generated_images_epoch_{epoch+1}_sample_{i}.png')

            # Early stopping based on generator loss
            if val_gen_loss < best_val_loss: # early stopping
                best_val_loss = val_gen_loss
                best_epoch = epoch
                patience_counter = 0  # Reset counter if the model improves
                self.save() # Save model weights
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}. No improvement for {self.patience} consecutive epochs.")
                print(f"Best validation loss: {best_val_loss} at epoch {best_epoch + 1}")
                break

        # Restore the best model
        self.generator = load_model('saved_model/generator'+self.save_path)
        self.discriminator = load_model('saved_model/discriminator'+self.save_path)

    def generate(self, X, noise, conditions):
        X = X.reshape(1, *X.shape)
        noise = noise.reshape(1, *noise.shape)
        conditions = conditions.reshape(1, *conditions.shape)
        return self.generator.predict([X, noise, conditions], verbose=0)

    def visualize_loss(self):
        # Plot losses
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))

        axes[0].plot(self.train_losses_gen, label='loss')
        axes[0].plot(self.val_losses_gen, label='Validation loss')
        axes[0].set_title('Generator Loss')
        axes[0].legend()
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')

        axes[1].plot(self.train_losses_disc, label='loss')
        axes[1].plot(self.val_losses_disc, label='Validation loss')
        axes[1].set_title('Discriminator Loss')
        axes[1].legend()
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')

        plt.show()

    def visualize_generator_loss_components(self):
        plt.figure(figsize=(8, 3))

        plt.plot(np.array(self.loss_weight) * self.wasserstein_losses, '--',
            label='Wasserstein loss', color='darkred')
        plt.plot(np.array(self.flow_weight) * self.flow_losses, '--',
            label='Optical flow loss', color='green')
        plt.plot(np.array(self.mse_weight) * self.mse_losses, '--',
            label='MSE loss', color='darkblue')
        plt.plot(np.array(self.color_weight) * self.color_losses, '--',
            label='Color loss', color='purple')

        plt.plot(np.array(self.loss_weight) * self.val_wasserstein_losses, '--',
            label='Validation Wasserstein loss', color='red')
        plt.plot(np.array(self.flow_weight) * self.val_flow_losses, '--',
            label='Validation Optical flow loss', color='lime')
        plt.plot(np.array(self.mse_weight) * self.val_mse_losses, '--',
            label='Validation MSE loss', color='blue')
        plt.plot(np.array(self.color_weight) * self.val_color_losses, '--',
            label='Validation Color loss', color='magenta')

        plt.title('Generator Loss Components')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.show()

    def visualize_generated_image(self, X, z, cX):
        generated_images = self.generate(X, z, cX)[0]
        output_channels = self.generator.output_shape[1]
        cmap = ['gray' if output_channels == 1 else None][0]

        n_images = self.len_input_seq + self.len_generated_seq
        fig, axes = plt.subplots(1, n_images, figsize=((1+n_images)*4, 4))

        # input images (real)
        for i in range(self.len_input_seq):
            img = X if n_images == 1 else X[i]
            if output_channels == 1:
                img = img.reshape(self.generator.input_dim[1], self.generator.input_dim[2])
            else:
                img = np.moveaxis(img, 0, -1)
            axes[i].imshow(img, cmap=cmap)
            axes[i].axis('off')
            axes[i].set_title('Input, Frame %d' % i)

        # generated images
        for i in range(self.len_generated_seq):
            k = i + self.len_input_seq
            img = generated_images if n_images == 1 else generated_images[i]
            with warnings.catch_warnings(): # no warning for clipping
                warnings.simplefilter("ignore")
                if output_channels == 1:
                    img = img.reshape(self.generator.output_dim[1], self.generator.output_dim[2])
                    axes[k].imshow(img, cmap=cmap)
                else:
                    img = np.moveaxis(img, 0, -1)
                    axes[k].imshow(img, cmap=cmap)
            axes[k].axis('off')
            axes[k].set_title('Generated, Frame %d' % (i+1) )

        plt.show()
    
    def save_generated_image(self, X, z, cX, path='generated_image.png'):
        generated_images = self.generate(X, z, cX)[0]
        output_channels = self.generator.output_shape[1]
        cmap = ['gray' if output_channels == 1 else None][0]

        n_images = self.len_input_seq + self.len_generated_seq
        fig, axes = plt.subplots(1, n_images, figsize=((1+n_images)*4, 4))

        for i in range(self.len_input_seq):
            img = X if n_images == 1 else X[i]
            if output_channels == 1:
                img = img.reshape(self.generator.input_dim[1], self.generator.input_dim[2])
            else:
                img = np.moveaxis(img, 0, -1)
            axes[i].imshow(img, cmap=cmap)
            axes[i].axis('off')
            axes[i].set_title('Input, Frame %d' % i)

        for i in range(self.len_generated_seq):
            k = i + self.len_input_seq
            img = generated_images if n_images == 1 else generated_images[i]
            with warnings.catch_warnings(): # no warning for clipping
                warnings.simplefilter("ignore")
                if output_channels == 1:
                    img = img.reshape(self.generator.output_dim[1], self.generator.output_dim[2])
                    axes[k].imshow(img, cmap=cmap)
                else:
                    img = np.moveaxis(img, 0, -1)
                    axes[k].imshow(img, cmap=cmap)
            axes[k].axis('off')
            axes[k].set_title('Generated, Frame %d' % (i+1) )

        plt.savefig(path)
        plt.close()

    def save(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            save_model(self.generator, 'saved_model/generator'+self.save_path)
            save_model(self.discriminator, 'saved_model/discriminator'+self.save_path)
