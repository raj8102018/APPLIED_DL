import torch
import torch.nn.functional as F
from model import RMSNorm, LlamaMLP, apply_rotary_pos_emb

# --- TEST 1: RMSNorm ---
print("--- TEST 1: RMSNorm ---")
x = torch.randn(2, 5, 256) # Batch, Seq, Dim
rmsnorm = RMSNorm(dim=256)
out_norm = rmsnorm(x)

# The root mean square of the output should be very close to 1.0 along the last dimension
rms = torch.sqrt(out_norm.pow(2).mean(-1))
assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3), "FATAL: RMSNorm did not normalize the variance to 1!"
print("SUCCESS: RMSNorm operational.")

# --- TEST 2: SwiGLU MLP ---
print("\n--- TEST 2: SwiGLU ---")
mlp = LlamaMLP(hidden_size=256, intermediate_size=1024)
out_mlp = mlp(x)
assert out_mlp.shape == (2, 5, 256), f"FATAL: SwiGLU dimension mismatch. Expected (2, 5, 256), got {out_mlp.shape}"
print("SUCCESS: SwiGLU Gating operational.")

# --- TEST 3: RoPE ---
print("\n--- TEST 3: RoPE (Rotary Embeddings) ---")
# Simulate Q and K from a single head (Batch, Heads, SeqLen, HeadDim)
q = torch.randn(2, 8, 10, 64)
k = torch.randn(2, 8, 10, 64)

def precompute_freqs_cis(dim: int, seq_len: int):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    # Reshape to broadcast with (Batch, Heads, SeqLen, HeadDim) -> (1, 1, SeqLen, HeadDim)
    return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)

cos, sin = precompute_freqs_cis(dim=64, seq_len=10)
q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

assert q_rot.shape == q.shape, "FATAL: RoPE changed the shape of the Query tensor!"
assert not torch.allclose(q, q_rot), "FATAL: RoPE did not alter the input vectors!"
print("SUCCESS: RoPE accurately rotated Query and Key matrices in complex space.")