# Conditional GANs
### Conditional GANs for Video Prediction - Visual model for robot control

This repository contains implementations of Conditional GANs (CGANs) and Autoencoders designed for video prediction tasks.
The goal of this project is to develop a visual prediction model to assist in robot control through video frame generation. Various generative models, such as WGANs, VAEs, and Conditional VAEs, are implemented to predict and generate future frames in a video sequence.


## Project Description and Results

This work presents an innovative approach for the automatic generation of video sequences using a conditional Generative Adversarial Network (GAN). Our model is designed to predict the subsequent frames of a video by integrating contextual information through a conditioning system.

The final model used is a Wasserstein GAN (WGAN), which improves stability by employing the Wasserstein distance as a measure of the difference between the generated and real data distributions. This distance provides smoother gradients, aiding the model in converging better than traditional GANs.

The generator consists of a transpose convolutional neural network that uses a recursive approach to generate future samples, employing the teacher forcing technique. The generator loss is a combination of several loss metrics, including Wasserstein Loss, Mean Squared Error (MSE), Optical Flow Loss, and Color Loss.

The discriminator features a series of convolutional layers that learn patterns from images, followed by fully connected layers for classification. It leverages dropout and Leaky ReLU activations for improved generalization and stable gradient flow during training, utilizing a sigmoid activation function to output a probability value between 0 and 1.

We conducted experiments on two datasets: one containing objects and one without. The results indicate that our model achieved satisfactory visual quality, demonstrating its effectiveness in generating coherent video sequences.


## Folder Structure

```
ConditionalGANs/
│
├── generated_images/                 # Directory for storing generated images
├── test_results/                     # Directory for storing GIFs generated for visualizing model predictions
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
├── .gitignore                        # Files to ignore in version control
├── main_wgan.py                      # Main file for training the WGAN model
├── requirements.txt                  # Python dependencies
│
├── test_no_obj.ipynb                 # Jupyter Notebook for testing the model on a dataset without objects
├── test_obj.ipynb                    # Jupyter Notebook for testing the model on a dataset with objects
├── train_no_obj.ipynb                # Jupyter Notebook for training using dataset without objects
├── train_obj.ipynb                   # Jupyter Notebook for training using dataset with objects
```

### Notebooks Overview

- **Training the Model with Custom Parameters**
  - **`train_no_obj.ipynb`** - This notebook demonstrates how to train the Conditional GAN model on a dataset without objects. You can visualize the training process and adjust hyperparameters to experiment with custom configurations if you want to train the model from scratch or fine-tune it further.
  - **`train_obj.ipynb`** - Similar to `train_no_obj.ipynb`, this notebook trains the model on a dataset containing background objects. It also allows for hyperparameter adjustments, providing flexibility to adapt the model to different scenarios.

- **Testing the Pretrained Model with Selected Parameters**
  - **`test_no_obj.ipynb`** - This notebook tests the pretrained model on a dataset without objects, using the best hyperparameter configuration selected from training. It includes visualization of generated frames and an option to recursively generate extended sequences.
  - **`test_obj.ipynb`** - Tests the pretrained model on the dataset containing objects, again using the optimal parameter configuration chosen. Visuals of generated frames versus true frames are provided for evaluation.

**To run these notebooks:**
Add the `saved_model` and `data` folders to `ConditionalGANs/ConditionalGANs`.
- `saved_model` is required only for testing notebooks.
- `data` is required for both training and testing notebooks.

If you need access to the `saved_model` and `data` folders, please feel free to contact me.


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
