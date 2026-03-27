import torch
import torch.nn.functional as F
from model import TiledAttention

# Hyperparameters
BATCH = 2
SEQ_LEN = 1024  # Large enough to test tiling
HEAD_DIM = 64
BLOCK_SIZE = 128

print("--- RUNNING TEST: FlashAttention Tiling Algorithm ---")

q = torch.randn(BATCH, SEQ_LEN, HEAD_DIM)
k = torch.randn(BATCH, SEQ_LEN, HEAD_DIM)
v = torch.randn(BATCH, SEQ_LEN, HEAD_DIM)

# 1. Standard Naive Attention (The Ground Truth)
scale = HEAD_DIM ** -0.5
scores = (q @ k.transpose(-2, -1)) * scale
attn_weights = F.softmax(scores, dim=-1)
out_naive = attn_weights @ v

# 2. Our Tiled Attention
tiled_attn = TiledAttention()
out_tiled = tiled_attn(q, k, v, block_size=BLOCK_SIZE)

# Assertions
diff = torch.abs(out_naive - out_tiled).max().item()
assert diff < 1e-4, f"FATAL: Tiled attention diverges from standard attention! Max diff: {diff}"

print("SUCCESS: Online Softmax Tiling accurately matches standard O(N^2) attention.")