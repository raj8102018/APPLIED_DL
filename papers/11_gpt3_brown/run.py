import torch
from model import KVCacheAttention, GPT

# ==========================================
# TEST 1: The Attention Equivalence Proof
# ==========================================
print("--- RUNNING TEST 1: KV Cache Mathematical Equivalence ---")

# Hyperparameters for Attention
BATCH_SIZE = 2
SEQ_LEN = 10
HIDDEN_SIZE = 256
NUM_HEADS = 8

# Dummy Data
x_full = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
attention = KVCacheAttention(hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS)

attention.eval()
with torch.no_grad():
    # --- THE NAIVE PASS (O(N^2) Recalculation) ---
    out_naive, _ = attention(x_full)
    target_logit = out_naive[:, -1, :] # We only care about the final output vector

    # --- THE KV CACHE PASS (Optimized) ---
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
print(f"New KV Cache shape (Keys): {new_kv[0].shape}\n")


# ==========================================
# TEST 2: The Stateful Generation Loop Proof
# ==========================================
print("--- RUNNING TEST 2: Autoregressive Stateful Generation ---")

# Hyperparameters for Full Model
VOCAB_SIZE = 50257
NUM_LAYERS = 2
MAX_POSITIONS = 1024

# Init Model
model = GPT(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    max_positions=MAX_POSITIONS,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=1024
)

model.eval()
# Dummy Prompt: Batch of 1, 5 tokens long
prompt = torch.randint(0, VOCAB_SIZE, (1, 5)) 

with torch.no_grad():
    print("Starting optimized generation...")
    # This will crash if your past_kv threading or position_ids are misaligned
    generated_sequence = model.generate(
        idx=prompt,
        max_new_tokens=10,
        temperature=0.8
    )

print(f"Original Prompt Shape: {prompt.shape}")
print(f"Generated Sequence Shape: {generated_sequence.shape}")
assert generated_sequence.shape == (1, 15), f"FATAL: Generation loop failed. Expected (1, 15), got {generated_sequence.shape}"
print("SUCCESS: GPT-3 KV Cache Generation Engine is fully operational.")