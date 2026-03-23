import torch
from model import MarginalizedMLMLoss

# Hyperparameters
BATCH_SIZE = 2
TOP_K = 3
VOCAB_SIZE = 100

loss_fn = MarginalizedMLMLoss()

# Dummy Data
retriever_logits = torch.randn(BATCH_SIZE, TOP_K, requires_grad=True)
generator_logits = torch.randn(BATCH_SIZE, TOP_K, VOCAB_SIZE, requires_grad=True)
target_tokens = torch.tensor([42, 88]) # The correct masked words

# Forward Pass
loss = loss_fn(retriever_logits, generator_logits, target_tokens)
print(f"Marginalized Loss: {loss.item():.4f}")

# Backward Pass Proof
loss.backward()

# The Assertions
assert retriever_logits.grad is not None, "FATAL: Gradients did not flow back to the Retriever!"
assert generator_logits.grad is not None, "FATAL: Gradients did not flow back to the Generator!"

print(f"Retriever Gradients Shape: {retriever_logits.grad.shape}")
print("SUCCESS: End-to-end differentiable retrieval is mathematically operational.")