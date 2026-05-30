# VAE — Kingma & Welling

## 3-Bullet Math Summary

- 
- 
- 

## Why did this fail/succeed?


# Auto-Encoding Variational Bayes (VAE)

**One-Line Impact:** Implemented a Variational Autoencoder (VAE) from scratch in PyTorch, successfully training on MNIST to map high-dimensional image data into a continuous, generative latent space.

## 📊 Proof of Work & Evaluation
This folder contains `evaluation.ipynb`, an end-to-end interactive training and evaluation walkthrough. 

* **Dataset:** MNIST (Flattened 28x28)
* **Architecture:** Fully Connected Encoder/Decoder with Reparameterization Trick (Latent Dim = 20)
* **Objective:** Binary Cross Entropy (Reconstruction) + KL Divergence (Regularization)
* **Training Dynamics:** Trained for 50 epochs using Adam. 
* **Results:** The notebook successfully demonstrates the separation of Total Loss, Reconstruction Loss, and KL Divergence over time. Qualitative evaluation proves the model's ability to sample random noise from the normal distribution and decode it into recognizable handwritten digits.

## 💻 Reproducibility
The `evaluation.ipynb` notebook is designed to be fully self-contained. 
1. Open the notebook in Jupyter or Google Colab.
2. Run all cells. 
3. The notebook will automatically define the architecture, download the dataset, execute the 50-epoch training loop, and plot the generated samples and loss curves.

## 🧠 Core Engineering Concepts Demonstrated
* **Reparameterization Trick:** Differentiable sampling for backpropagation through stochastic nodes.
* **Loss Function Design:** Balancing generative reconstruction quality with latent space regularization.