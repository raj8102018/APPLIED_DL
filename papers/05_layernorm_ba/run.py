import torch
import torch.nn as nn
from model import CustomLayerNorm

# Hyperparameters
BATCH_SIZE = 32
SEQ_LEN = 50
HIDDEN_DIM = 512

# Dummy Data
x_custom = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
x_official = x_custom.clone().detach().requires_grad_(True)

# Init Modules
custom_ln = CustomLayerNorm(HIDDEN_DIM)
official_ln = nn.LayerNorm(HIDDEN_DIM)

# --- FORWARD PASS TEST ---
out_custom = custom_ln(x_custom)
out_official = official_ln(x_official)

forward_diff = torch.abs(out_custom - out_official).max().item()
assert forward_diff < 1e-5, f"FATAL: Forward pass divergence! Max diff: {forward_diff}"
print("SUCCESS: Forward pass matches official PyTorch implementation.")

# --- BACKWARD PASS TEST ---
# Create a dummy loss by summing all elements and backpropagate
out_custom.sum().backward()
out_official.sum().backward()

# Compare gradients on the learnable parameters
gamma_diff = torch.abs(custom_ln.gamma.grad - official_ln.weight.grad).max().item()
beta_diff = torch.abs(custom_ln.beta.grad - official_ln.bias.grad).max().item()

assert gamma_diff < 1e-4, f"FATAL: Gamma gradient divergence! Max diff: {gamma_diff}"
assert beta_diff < 1e-4, f"FATAL: Beta gradient divergence! Max diff: {beta_diff}"
print("SUCCESS: Backward pass and parameter gradients are perfectly aligned.")