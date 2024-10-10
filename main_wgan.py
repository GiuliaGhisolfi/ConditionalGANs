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
        patience=3,
        learning_rate_generator=0.005,
        learning_rate_discriminator=0.002,
        n_critic=3,  # number of critic updates per generator update
        clip_value=0.015,
        discriminator_gradient_penalty_weight=0.02,
        generator_wloss_weight=0, #0.1,
        generator_flow_loss_weight=0, #0.4,
        generator_mse_weight=1
    )

    gan.train(
        X_train_z[:500], X_train[:500], cX_train[:500], # noise, real images, conditions
        X_val_z[:500], X_val[:500], cX_val[:500],
        epochs=2, batch_size=200
    )

if __name__ == '__main__':
    main()