import torch
import torch.nn as nn
from model import ResidualBlock

# Dummy Data: [batch_size, channels, height, width]
# Assuming a standard 64x64 feature map input
x = torch.randn(16, 64, 64, 64, requires_grad=True)

# Test 1: Identity Block (No stride, channels stay the same)
identity_block = ResidualBlock(in_channels=64, out_channels=64, stride=1)
out_identity = identity_block(x)

assert out_identity.shape == (16, 64, 64, 64), f"Identity spatial mismatch: {out_identity.shape}"
print("SUCCESS: Identity Block tensor geometry perfectly maintained.")

# Test 2: Projection Block (Stride 2, channels double to 128)
# This forces spatial dimensions to halve (64x64 -> 32x32)
projection_block = ResidualBlock(in_channels=64, out_channels=128, stride=2)
out_projection = projection_block(x)

assert out_projection.shape == (16, 128, 32, 32), f"Projection spatial mismatch: {out_projection.shape}"
print("SUCCESS: Projection Block tensor geometry perfectly halved and doubled.")

# Test 3: Gradient Flow
out_projection.sum().backward()
assert x.grad is not None, "FATAL: Gradients failed to flow through the skip connection!"
print("SUCCESS: Gradients are successfully flowing through the residual architecture.")