from tensorflow.keras.models import load_model

from src.utils.gan_dataloader import get_gan_dataset
from src.utils.make_discriminator import make_discriminator
from src.utils.make_generator import make_generator
from src.wgan import WassersteinGAN


def get_data():
    import pickle

    with open('data/new_data_no_obj.pkl', 'rb') as f:
        X_train, cX_train, X_train_z = pickle.load(f)

    # split the training set into training and validation
    n_sample_train = int(X_train.shape[0] * 0.8)

    X_train, X_val = X_train[:n_sample_train], X_train[n_sample_train:]
    cX_train, cX_val = cX_train[:n_sample_train], cX_train[n_sample_train:]
    X_train_z, X_val_z = X_train_z[:n_sample_train], X_train_z[n_sample_train:]

    len_input_seq = X_train.shape[1] # 6
    len_generated_seq = 4

    return X_train, X_val, X_train_z, X_val_z, cX_train, cX_val, len_input_seq, len_generated_seq

def main():
    X_train, X_val, X_train_z, X_val_z, cX_train, cX_val, len_input_seq, len_generated_seq = get_data()

    generator = make_generator(
        image_dim=[3, 64, 64],
        latent_dim=1280,
        conditions_dim=6,
        len_input_seq=len_input_seq,
        len_output_seq=len_generated_seq,
        n_filters=[16, 8, 8],
        kernel_size=[5, 3, 1],
        stride=[1, 1, 1],
        padding=['same', 'same', 'same'],
        hidden_dims=[128, 128],
    )

    discriminator = make_discriminator(
        input_dim=[3, 64, 64],
        len_input_seq=min(len_input_seq, len_generated_seq),
    )

    gan = WassersteinGAN(
        generator=generator,
        discriminator=discriminator,
        len_input_seq=len_input_seq,
        len_generated_seq=len_generated_seq,
        patience=5,
        learning_rate_generator=0.01,
        learning_rate_discriminator=0.01,
        n_critic=5,  # number of critic updates per generator update
        clip_value=0.01,
        discriminator_gradient_penalty_weight=0.02,
        generator_wloss_weight=0.5,
        generator_flow_loss_weight=0.3,
        generator_mse_weight=0.15,
        color_weight=0.05,
        save_path = 'wgan_train_obj'
    )

    gan.train(
        X_train_z, X_train, cX_train, # noise, real images, conditions
        X_val_z, X_val, cX_val,
        epochs=20, batch_size=1
    )

if __name__ == '__main__':
    main()