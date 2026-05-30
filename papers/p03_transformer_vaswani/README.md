# Transformer — Vaswani et al.

## 3-Bullet Math Summary

- 
- 
- 

## Why did this fail/succeed?


# Attention Is All You Need (Transformer)

**One-Line Impact:** Implemented the canonical sequence-to-sequence Transformer architecture, verifying attention mechanics and causal masking via a synthetic autoregressive copy-task.

## 📊 Proof of Work & Evaluation
This folder contains `evaluation.ipynb`, an interactive experimental notebook demonstrating the Transformer's learning capabilities on a controlled synthetic task.

* **Dataset:** Synthetic Sequence Copy-Task.
* **Architecture:** Full Encoder-Decoder Transformer with Multi-Head Attention, Scaled Dot-Product Attention, and Sinusoidal Positional Encodings.
* **Objective:** Cross-Entropy Loss (Token prediction).
* **Training Dynamics:** Includes an 80/20 Train-Validation split to monitor generalization.
* **Results:** Rather than translating text, this notebook proves the fundamental *algorithmic routing* of the Transformer. By training it on a synthetic copy-task, it explicitly verifies that the causal masking (`tril`), positional encodings, and gradient flows are mathematically sound before scaling to massive NLP datasets.

## 💻 Reproducibility
The `evaluation.ipynb` notebook is self-contained. 
1. Open the notebook in Jupyter or Google Colab.
2. Run all cells to initialize the sequence datasets, build the causal masks, execute the training/validation loops, and plot the loss curves.

## 🧠 Core Engineering Concepts Demonstrated
* **Attention Mechanics:** Manual implementation of `Q, K, V` matrix multiplications and softmax scaling.
* **Autoregressive Masking:** Implementation of lower-triangular masks to prevent look-ahead data leakage during sequence generation.
* **Validation Infrastructure:** Isolating train/val splits to track generalization in sequence models.