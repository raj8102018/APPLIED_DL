# GAN — Goodfellow et al.

## 3-Bullet Math Summary

- 
- 
- 

## Why did this fail/succeed?


# Generative Adversarial Networks (GAN)

**One-Line Impact:** Engineered a foundational GAN architecture with alternating adversarial training loops, successfully mapping random noise to coherent MNIST digits.

## 📊 Proof of Work & Evaluation
This folder contains `evaluation.ipynb`, an end-to-end interactive training and evaluation walkthrough. 

* **Dataset:** MNIST (Normalized to `[-1, 1]`)
* **Architecture:** Fully Connected Generator (Tanh output) and Discriminator (Sigmoid output). Latent Dim = 100.
* **Objective:** Minimax Binary Cross-Entropy Loss.
* **Training Dynamics:** Trained for 50 epochs. Features decoupled Adam optimizers for the Generator and Discriminator, with a carefully tuned learning rate disparity (`lr_D = lr_G / 2`) to prevent the discriminator from overpowering the generator early in training.
* **Results:** The notebook logs the adversarial loss dynamics step-by-step. Qualitative evaluation displays the generator's progression from random static to structured digits over the training lifecycle.

## 💻 Reproducibility
The `evaluation.ipynb` notebook is fully self-contained for easy verification.
1. Open the notebook in Jupyter or Google Colab.
2. Run all cells. The notebook will define the models, handle data normalization, execute the alternating training loop, and visualize the generated batches.

## 🧠 Core Engineering Concepts Demonstrated
* **Adversarial Optimization:** Managing unstable, non-convex minimax games with alternating gradient steps.
* **Activation Bounding:** Strictly managing tensor bounds (Tanh/Sigmoid) to prevent gradient explosion.