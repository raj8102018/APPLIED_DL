import torch
from model import SwitchMoELayer

# Hyperparameters
BATCH_SIZE = 2
SEQ_LEN = 10
HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 256
NUM_EXPERTS = 4

print("--- RUNNING TEST: Switch Transformer MoE Routing ---")

# Init Model
moe_layer = SwitchMoELayer(
    hidden_size=HIDDEN_SIZE, 
    intermediate_size=INTERMEDIATE_SIZE, 
    num_experts=NUM_EXPERTS
)

# Dummy Input
x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)

# Forward Pass
out = moe_layer(x)

# Assertions
print(f"Input Shape: {x.shape}")
print(f"Output Shape: {out.shape}")

assert out.shape == x.shape, f"FATAL: MoE layer changed the tensor shape! Expected {x.shape}, got {out.shape}"

# A quick check to ensure gradients flow through the router
out.sum().backward()
assert moe_layer.router.weight.grad is not None, "FATAL: Gradients are not flowing back through the Router!"

print("SUCCESS: Sparse Mixture of Experts routing is fully operational.")