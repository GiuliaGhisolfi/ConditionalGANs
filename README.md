# ConditionalGANs
Conditional GANs for Video Prediction - Visual model for robot control

## Folder Structure

```
ConditionalGANs/
│
├── generated_images/       # Directory for storing generated images
│
├── src/
│   ├── autoencoders/
│   │   ├── IPvae.py                  # Implementation of a Identity Preserving VAE
│   │   ├── ae.py                     # Implementation of an Autoencoder (AE)
│   │   ├── cvae.py                   # Implementation of a Conditional VAE (CVAE)
│   │   ├── vae.py                    # Implementation of a Variational Autoencoder (VAE)
│   ├── generator.py                  # Generator class for the latent approximation
│   ├── latent_vector_approximator.py # Code for approximating latent vectors
│   ├── wgan.py                       # Wasserstein GAN (WGAN) implementation
│
├── .gitignore                         # Files to ignore in version control
├── main_wgan.py                       # Main file for training the WGAN model
├── requirements.txt                   # Python dependencies
├── train_no_obj.ipynb                 # Jupyter Notebook for training using dataset without objects
├── train_obj.ipynb                    # Jupyter Notebook for training using dataset with objects
```

## Setting up the environment
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ConditionalGANs.git
    cd ConditionalGANs
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
