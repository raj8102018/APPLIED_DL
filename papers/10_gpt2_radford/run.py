import torch
import torch.nn.functional as F
# Import your updated GPT model from wherever you saved it
from model import GPT 

# Hyperparameters
VOCAB_SIZE = 50257 # GPT-2's BPE vocab size
HIDDEN_SIZE = 768
NUM_LAYERS = 4

# Init Model
model = GPT(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    max_positions=1024,
    num_layers=NUM_LAYERS,
    num_heads=12,
    d_ff=3072
)

# Dummy Prompt: Shape [1, 5] (Batch of 1, 5 tokens long)
prompt = torch.randint(0, VOCAB_SIZE, (1, 5))

# Generation Test
model.eval() # MUST be in eval mode for inference
with torch.no_grad():
    generated_sequence = model.generate(
        idx=prompt,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50
    )

# The Assertions
print(f"Original Prompt Shape: {prompt.shape}")
print(f"Generated Sequence Shape: {generated_sequence.shape}")

assert generated_sequence.shape == (1, 25), f"FATAL: Generation loop failed to append tokens correctly. Expected (1, 25), got {generated_sequence.shape}"
print("SUCCESS: GPT-2 Autoregressive Generation Engine is fully operational.")