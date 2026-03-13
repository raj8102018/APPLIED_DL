import torch
from model import CustomAlexNet

# Hyperparameters
BATCH_SIZE = 4
CHANNELS = 3
HEIGHT = 224
WIDTH = 224
NUM_CLASSES = 1000

# Dummy Data representing a batch of 4 RGB images
x = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)

# Init Model
model = CustomAlexNet(num_classes=NUM_CLASSES)

# Forward Pass
logits = model(x)

# The Assertion
print(f"Input Shape: {x.shape}")
print(f"Output Shape: {logits.shape}")

assert logits.shape == (BATCH_SIZE, NUM_CLASSES), f"FATAL: Dimension mismatch! Expected {(BATCH_SIZE, NUM_CLASSES)}, got {logits.shape}"
print("SUCCESS: AlexNet forward pass and dimensional reduction are mathematically sound.")