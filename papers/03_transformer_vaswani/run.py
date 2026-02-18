from attention import ScaledDotProductAttention, MultiHeadAttention
from model import EncoderBlock, Transformer # (or wherever you put it)
import torch
Q =torch.tensor([[[[1, 0],
   [0, 1]]]], dtype=torch.float32)
K =torch.tensor([[[[1, 0],
   [0, 1]]]],dtype=torch.float32)
V =torch.tensor([[[[10, 0],
   [0, 20]]]],dtype=torch.float32)

value = ScaledDotProductAttention(2)
value=value.forward(Q,K,V)
print(value)

# HYPERPARAMETERS
BATCH_SIZE = 64
SEQ_LEN = 50
D_MODEL = 512
NUM_HEADS = 8
HEAD_DIM = D_MODEL // NUM_HEADS # 64

# INPUTS
x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL) # Simulation of embeddings
mask = torch.ones(BATCH_SIZE, 1, 1, SEQ_LEN) # Dummy mask

# INIT MODEL
mha = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS)

# FORWARD PASS
output = mha(x, x, x, mask=mask) # Self-attention: Q=K=V=x

# THE ASSERTION
print(f"Input Shape: {x.shape}")
print(f"Output Shape: {output.shape}")

assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL), "FATAL: Output shape mismatch!"
print("SUCCESS: Multi-Head Attention tensor shapes are correct.")

# --- ENCODER BLOCK TEST ---

encoder_block = EncoderBlock(d_model=512, num_heads=8, d_ff=2048, dropout=0.1)

# We use the same 'x' and 'mask' from the previous test
enc_out = encoder_block(x, mask)

print(f"Encoder Block Output Shape: {enc_out.shape}")
assert enc_out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL), "FATAL: Encoder Block shape mismatch!"
print("SUCCESS: Encoder Block routing is correct.")

# --- FULL TRANSFORMER TEST ---

# Hyperparameters
SRC_VOCAB_SIZE = 5000
TGT_VOCAB_SIZE = 5000
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 6
D_FF = 2048
MAX_SEQ_LEN = 100
DROPOUT = 0.1
BATCH_SIZE = 32

# Dummy Token Data (integers, not floats)
src_data = torch.randint(0, SRC_VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))
tgt_data = torch.randint(0, TGT_VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))

# Dummy Masks
src_mask = torch.ones(BATCH_SIZE, 1, 1, MAX_SEQ_LEN)
tgt_mask = torch.tril(torch.ones(1, 1, MAX_SEQ_LEN, MAX_SEQ_LEN))

# Init Model
transformer = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, DROPOUT, MAX_SEQ_LEN)

# Forward Pass
logits = transformer(src_data, tgt_data, src_mask, tgt_mask)

print(f"Final Logits Shape: {logits.shape}")
assert logits.shape == (BATCH_SIZE, MAX_SEQ_LEN, TGT_VOCAB_SIZE), "FATAL: Transformer output shape mismatch!"
print("SUCCESS: The Vaswani Transformer is fully operational.")