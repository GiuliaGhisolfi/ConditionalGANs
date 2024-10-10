import pickle

import numpy as np

from src.utils.utils import read_latent_vectors


def get_gan_dataset(len_input_seq=2, scenario='obj'): # FIXME: scenario is not used +all in one pickle file
    X_train_z, _ = read_latent_vectors('predictions/no_obj/z_vae_rgb_train.pkl')
    X_test_z, _ = read_latent_vectors('predictions/no_obj/z_vae_rgb_val.pkl')

    Y_train_z, _ = read_latent_vectors('predictions/no_obj/zY_vae_rgb_train.pkl')
    Y_test_z, _ = read_latent_vectors('predictions/no_obj/zY_vae_rgb_val.pkl')
    
    X_train = pickle.load(open('predictions/no_obj/X_train.pkl', 'rb'))
    X_test = pickle.load(open('predictions/no_obj/X_val.pkl', 'rb'))
    Y_train = pickle.load(open('predictions/no_obj/Y_train.pkl', 'rb'))
    Y_test = pickle.load(open('predictions/no_obj/Y_val.pkl', 'rb'))
    cX_train = pickle.load(open('predictions/no_obj/cX_train.pkl', 'rb'))
    cX_test = pickle.load(open('predictions/no_obj/cX_val.pkl', 'rb'))

    # split the training set into training and validation
    n_sample_train = int(X_train.shape[0] * 0.8)

    X_train, X_val = X_train[:n_sample_train], X_train[n_sample_train:]
    Y_train, Y_val = Y_train[:n_sample_train], Y_train[n_sample_train:]
    cX_train, cX_val = cX_train[:n_sample_train], cX_train[n_sample_train:]
    X_train_z, X_val_z = X_train_z[:n_sample_train], X_train_z[n_sample_train:]
    Y_train_z, Y_val_z = Y_train_z[:n_sample_train], Y_train_z[n_sample_train:]

    if len_input_seq == 2:
        # Concatenate each element of the zX with zY
        X_train_z = np.array([np.concatenate(([X_train_z[i]], [Y_train_z[i]])) for i in range(len(X_train_z))])
        X_val_z = np.array([np.concatenate(([X_val_z[i]], [Y_val_z[i]])) for i in range(len(X_val_z))])
        X_test_z = np.array([np.concatenate(([X_test_z[i]], [Y_test_z[i]])) for i in range(len(X_test_z))])

        # Concatenate each element of the X with Y
        X_train = np.array([[X_train[i], Y_train[i]] for i in range(len(X_train))])
        X_val = np.array([[X_val[i], Y_val[i]] for i in range(len(X_val))])
        X_test = np.array([[X_test[i], Y_test[i]] for i in range(len(X_test))])

        #cX_train = np.array([np.concatenate(([cX_train[i]], [cX_train[i]])) for i in range(len(cX_train))])
        #cX_val = np.array([np.concatenate(([cX_val[i]], [cX_val[i]])) for i in range(len(cX_val))])
        #cX_test = np.array([np.concatenate(([cX_test[i]], [cX_test[i]])) for i in range(len(cX_test))])
    
    cX_train =  np.reshape(cX_train, (cX_train.shape[0], 1, cX_train.shape[1]))
    cX_val =  np.reshape(cX_val, (cX_val.shape[0], 1, cX_val.shape[1]))
    cX_test =  np.reshape(cX_test, (cX_test.shape[0], 1, cX_test.shape[1]))

    return X_train, X_val, X_test, X_train_z, X_val_z, X_test_z, cX_train, cX_val, cX_test