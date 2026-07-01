# Deep Learning Architecture & Systems: A Code-First Study

This repository is a self-directed, code-first study of 49 foundational papers in Deep Learning, Large Language Models (LLMs), and AI Systems.

The objective is to bridge the gap between theoretical understanding and systems engineering by building the mathematical and architectural backbone of modern AI from scratch, primarily using pure PyTorch.

## 📊 Current Repository Status
* **Paper Map Total:** 49 Papers
* **Current Repo Snapshot:** 27 Paper Folders
* **PyTorch Implementations:** 19 Folders
* **Systems/Theory Notes:** 8 Folders

**Legend:**
* `[✓]` **Implemented:** Core architecture implemented in PyTorch (`model.py`), verification scripts (`run.py`), and study notes (`notes.pdf`).
* `[📝]` **Read Only / Systems:** Pure hardware/systems innovations containing detailed study notes, but no PyTorch implementation.
* `[ ]` **Planned:** On the roadmap but not yet present in the repository.

*Note on Numbering: Folder numbering (e.g., `01_vae`, `23_flashattention`) strictly follows my chronological reading order. Any gaps in the sequence represent papers currently in progress.*

---

## 🗺️ The 8-Domain Paper Map

### Domain 1: Deep Learning Foundations
* `[✓]` Auto-Encoding Variational Bayes (VAE)
* `[✓]` Generative Adversarial Nets (GAN)
* `[✓]` ADAM: A Method for Stochastic Optimization
* `[✓]` Layer Normalization
* `[✓]` Deep Residual Learning for Image Recognition (ResNet)
* `[✓]` ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)
* `[✓]` Attention is All You Need
* `[ ]` Decoupled Weight Decay Regularization

### Domain 2: The LLM Scaling & Alignment Era
* `[✓]` BERT: Pre-training of Deep Bidirectional Transformers
* `[✓]` Improving Language Understanding by Generative Pre-Training (GPT-1)
* `[✓]` Language Models are Unsupervised Multitask Learners (GPT-2)
* `[✓]` Language Models are Few-Shot Learners (GPT-3)
* `[📝]` Scaling Laws for Neural Language Models
* `[📝]` Training Compute-Optimal Large Language Models (Chinchilla)
* `[✓]` InstructGPT / Training Language Models to Follow Instructions
* `[ ]` Reinforcement Learning from Human Feedback
* `[📝]` Self-Instruct: Aligning Language Models with Self-Generated Instructions

### Domain 3: Modern Architecture & Retrieval
* `[✓]` Retrieval-Augmented Generation (RAG)
* `[✓]` REALM: Retrieval-Augmented Language Model Pre-Training
* `[✓]` LLaMA: Open and Efficient Foundation Models
* `[✓]` Switch Transformers: Scaling to Trillion Parameter Models
* `[✓]` FlashAttention: Fast and Memory-Efficient Exact Attention
* `[📝]` FlashAttention-2: Faster Attention with Better Parallelism 
* `[ ]` PaLM: Pathways Language Model 

### Domain 4: Inference, Serving & Systems
* `[📝]` Fast Inference from Transformers via Speculative Decoding
* `[📝]` Efficient Memory Management for LLM Serving with PagedAttention
* `[ ]` ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

### Domain 5: Quantization & Parameter-Efficient Fine-Tuning
* `[✓]` LoRA: Low-Rank Adaptation of Large Language Models
* `[📝]` Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (Jacob et al.)
* `[📝]` LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
* `[ ]` AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning
* `[ ]` Parameter-Efficient Fine-Tuning Methods: A Critical Review
* `[ ]` GPTQ: Accurate Post-Training Quantization for GPT
* `[ ]` AWQ: Activation-Aware Weight Quantization for LLMs

### Domain 6: Modern Vision & Self-Supervised Learning
* `[ ]` ViT: An Image is Worth 16x16 Words
* `[ ]` SimCLR: A Simple Framework for Contrastive Learning
* `[ ]` MoCo: Momentum Contrast for Unsupervised Visual Representation
* `[ ]` Self-Supervised Models are Continual Learners
* `[ ]` BAYPRANOMETA: Bayesian Proto-MAML for Few-Shot Anomaly Detection

### Domain 7: Deep Reinforcement Learning
* `[✓]` Proximal Policy Optimization Algorithms (PPO)
* `[ ]` Soft Actor-Critic: Off-Policy Maximum Entropy RL
* `[ ]` Addressing Function Approximation Error in Actor-Critic Methods

### Domain 8: Advanced Generative & AI Philosophy
* `[ ]` Denoising Diffusion Probabilistic Models (DDPM)
* `[ ]` High-Resolution Image Synthesis with Latent Diffusion Models
* `[ ]` A Survey on Generative Adversarial Networks
* `[ ]` Deep Generative Modelling: A Comparative Review
* `[ ]` Holistic Evaluation of Language Models
* `[ ]` Interpretable Machine Learning: Principles & Grand Challenges
* `[ ]` The Bitter Lesson