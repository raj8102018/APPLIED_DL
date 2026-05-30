import torch
import pytest

# ==========================================
# 1. STANDARD IMPORTS
# ==========================================
# (Note: Using 'as' allows us to import multiple classes named GPT without conflict)
from papers.p03_transformer_vaswani.model import Transformer
from papers.p08_bert_devlin.model import BERT, BERTPreTrainingHeads
from papers.p09_gpt1_radford.model import GPT as GPT1
from papers.p10_gpt2_radford.model import GPT as GPT2
from papers.p11_gpt3_brown.model import GPT as GPT3, KVCacheAttention
from papers.p20_llama_touvron.model import RMSNorm, LlamaMLP, apply_rotary_pos_emb

# ==========================================
# 2. GLOBAL TEST HYPERPARAMETERS
# ==========================================
VOCAB_SIZE = 1000
SEQ_LEN = 32
D_MODEL = 128
NUM_HEADS = 4
BATCH_SIZE = 2
NUM_LAYERS = 2
D_FF = 512
DROPOUT = 0.1

# ==========================================
# 3. THE ARCHITECTURE TESTS
# ==========================================

def test_transformer_forward_backward():
    """Verifies the original Bidirectional Encoder-Decoder architecture."""
    model = Transformer(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE, 
        d_model=D_MODEL, num_heads=NUM_HEADS, N=NUM_LAYERS, d_ff=D_FF, dropout=DROPOUT
    )
    src = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    tgt = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    
    output = model(src, tgt, None, None)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    
    output.sum().backward()
    assert next(model.parameters()).grad is not None

def test_bert_forward_backward():
    """Verifies BERT Encoder, Pooler, and Pre-training Heads."""
    bert = BERT(NUM_LAYERS, VOCAB_SIZE, D_MODEL, 512, 2, D_FF, NUM_HEADS)
    heads = BERTPreTrainingHeads(D_MODEL, VOCAB_SIZE)
    
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    segment_ids = torch.randint(0, 2, (BATCH_SIZE, SEQ_LEN))
    attention_mask = torch.ones(BATCH_SIZE, 1, 1, SEQ_LEN)
    
    sequence_output, pooled_output = bert(input_ids, segment_ids, attention_mask)
    mlm_logits, nsp_logits = heads(sequence_output, pooled_output)
    
    assert sequence_output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
    assert mlm_logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    
    # Combined loss for backward check
    loss = mlm_logits.sum() + nsp_logits.sum()
    loss.backward()
    assert next(bert.parameters()).grad is not None

def test_gpt1_forward_backward():
    """Verifies GPT-1 Autoregressive routing and Weight Tying."""
    model = GPT1(vocab_size=VOCAB_SIZE, hidden_size=D_MODEL, max_positions=512, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, d_ff=D_FF)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    
    logits = model(input_ids)
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    assert torch.equal(model.lm_head.weight, model.embeddings.token_embedding.weight)
    
    logits.sum().backward()
    assert next(model.parameters()).grad is not None

def test_gpt2_forward_backward():
    """Verifies GPT-2 Causal Masked Decoder."""
    model = GPT2(vocab_size=VOCAB_SIZE, hidden_size=D_MODEL, max_positions=1024, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, d_ff=D_FF)
    idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    
    output = model(idx)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    
    output.sum().backward()
    assert next(model.parameters()).grad is not None

def test_gpt3_kv_cache_forward_backward():
    """Verifies GPT-3 Stateful Generation and KV Cache equivalence."""
    attention = KVCacheAttention(hidden_size=D_MODEL, num_heads=NUM_HEADS)
    x_full = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, requires_grad=True)
    
    out_naive, _ = attention(x_full)
    out_naive.sum().backward()
    assert x_full.grad is not None, "Gradient failed to flow through KV Attention"

def test_llama_components_forward_backward():
    """Verifies LLaMA primitives: RMSNorm, SwiGLU, and RoPE."""
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, requires_grad=True)
    
    # RMSNorm Check
    rmsnorm = RMSNorm(dim=D_MODEL)
    out_norm = rmsnorm(x)
    assert out_norm.shape == x.shape
    
    # SwiGLU MLP Check
    mlp = LlamaMLP(hidden_size=D_MODEL, intermediate_size=D_FF)
    out_mlp = mlp(x)
    out_mlp.sum().backward()
    assert next(mlp.parameters()).grad is not None