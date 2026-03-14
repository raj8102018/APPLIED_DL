import torch
from model import BERT, BERTPreTrainingHeads

# Hyperparameters
BATCH_SIZE = 8
SEQ_LEN = 128
VOCAB_SIZE = 30000
HIDDEN_SIZE = 768
NUM_LAYERS = 2
NUM_HEADS=12
D_FF = 3072# Keep it small for the test

# Dummy Data
input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
segment_ids = torch.randint(0, 2, (BATCH_SIZE, SEQ_LEN)) # 0 for Sent A, 1 for Sent B
attention_mask = torch.ones(BATCH_SIZE, 1, 1, SEQ_LEN) # Dummy mask for EncoderBlock

# Init Models
# bert = BERT(VOCAB_SIZE, HIDDEN_SIZE, max_position_embeddings=512, type_vocab_size=2, num_layers=NUM_LAYERS)
bert = BERT(NUM_LAYERS, VOCAB_SIZE, HIDDEN_SIZE, 512, 2, D_FF, NUM_HEADS)
heads = BERTPreTrainingHeads(HIDDEN_SIZE, VOCAB_SIZE)

# Forward Pass
sequence_output, pooled_output = bert(input_ids, segment_ids, attention_mask)
mlm_logits, nsp_logits = heads(sequence_output, pooled_output)

# The Assertions
assert sequence_output.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE), f"Sequence output shape mismatch: {sequence_output.shape}"
assert pooled_output.shape == (BATCH_SIZE, HIDDEN_SIZE), f"Pooled output shape mismatch: {pooled_output.shape}"
assert mlm_logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), f"MLM Logits shape mismatch: {mlm_logits.shape}"
assert nsp_logits.shape == (BATCH_SIZE, 2), f"NSP Logits shape mismatch: {nsp_logits.shape}"

print("SUCCESS: BERT Embeddings, Pooler, and Pre-training Heads are perfectly routed.")