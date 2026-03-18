import torch
import torch.nn as nn
from model import LoRALinear

# Hyperparameters
IN_FEATURES = 4096
OUT_FEATURES = 4096
RANK = 8
BATCH_SIZE = 4
SEQ_LEN = 128

# Dummy Data
x = torch.randn(BATCH_SIZE, SEQ_LEN, IN_FEATURES)

# 1. Instantiate a massive standard layer
standard_layer = nn.Linear(IN_FEATURES, OUT_FEATURES)
# Clone the weights exactly so we can compare
original_weights = standard_layer.weight.clone().detach()

# 2. Get the baseline output
standard_layer.eval()
with torch.no_grad():
    baseline_out = standard_layer(x)

# 3. Wrap it in LoRA
lora_layer = LoRALinear(standard_layer, r=RANK)

# --- TEST 1: Identity Initialization ---
lora_layer.eval()
with torch.no_grad():
    lora_out = lora_layer(x)

diff = torch.abs(baseline_out - lora_out).max().item()
assert diff < 1e-5, f"FATAL: LoRA B matrix not initialized to zero! Max diff: {diff}"
print("SUCCESS: LoRA identity initialization is mathematically sound.")

# --- TEST 2: Parameter Count Compression ---
total_params = sum(p.numel() for p in lora_layer.parameters())
trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)

# The original layer has 4096 * 4096 + 4096 = 16,781,312 parameters.
# LoRA (A + B) has (4096 * 8) + (8 * 4096) = 65,536 parameters.
compression_ratio = total_params / trainable_params
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Compression Ratio: {compression_ratio:.2f}x reduction in gradient memory.")

assert trainable_params == 65536, "FATAL: Trainable parameter count is incorrect."
assert not lora_layer.original_layer.weight.requires_grad, "FATAL: Original layer was not frozen!"
print("SUCCESS: LoRA parameter freezing and rank decomposition are fully operational.")