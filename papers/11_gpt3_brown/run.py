import torch
from papers.11_gpt3_brown.model import KVCacheAttention

# Hyperparameters
BATCH_SIZE = 2
SEQ_LEN = 10
HIDDEN_SIZE = 256
NUM_HEADS = 8

# Dummy Data
x_full = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
attention = KVCacheAttention(hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS)

# --- THE NAIVE PASS (O(N^2) Recalculation) ---
attention.eval()
with torch.no_grad():
    # Pass all 10 tokens at once
    out_naive, _ = attention(x_full)
    target_logit = out_naive[:, -1, :] # We only care about the final output vector

# --- THE KV CACHE PASS (Optimized) ---
with torch.no_grad():
    # Step 1: Process the first 9 tokens and grab the cache
    x_past = x_full[:, :9, :]
    out_past, past_kv = attention(x_past)
    
    # Step 2: Process ONLY the 10th token, but pass in the cache
    x_current = x_full[:, 9:10, :] # Shape: [BATCH, 1, HIDDEN]
    out_cached, new_kv = attention(x_current, past_kv=past_kv)
    
    # The output of this single token should perfectly match the naive final token
    cached_logit = out_cached[:, -1, :]

# The Assertion
diff = torch.abs(target_logit - cached_logit).max().item()
assert diff < 1e-5, f"FATAL: KV Cache logic diverges from naive calculation! Max diff: {diff}"

print(f"SUCCESS: KV Cache Attention is mathematically equivalent to naive attention.")
print(f"New KV Cache shape (Keys): {new_kv[0].shape} (Target: [{BATCH_SIZE}, {NUM_HEADS}, {SEQ_LEN}, {HIDDEN_SIZE // NUM_HEADS}])")