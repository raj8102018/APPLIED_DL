import torch
from model import GPT

# Hyperparameters
BATCH_SIZE = 4
SEQ_LEN = 64
VOCAB_SIZE = 40000
HIDDEN_SIZE = 768
NUM_LAYERS = 4
NUM_HEADS = 12
D_FF = 3072

# Dummy Data (e.g., a batch of tokenized sentences)
input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

# Init Model
model = GPT(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    max_positions=512,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF
)

# Forward Pass
logits = model(input_ids)

# The Assertions
print(f"Input Shape: {input_ids.shape}")
print(f"Logits Shape: {logits.shape}")

assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), f"FATAL: Logits shape mismatch. Expected {(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)}, got {logits.shape}"

# Weight Tying Check
assert torch.equal(model.lm_head.weight, model.embeddings.token_embedding.weight), "FATAL: Weight tying failed! LM head and Token Embeddings must share memory."

print("SUCCESS: GPT architecture and Autoregressive Head are correctly routed.")