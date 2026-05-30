# test_advanced.py
import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# ADVANCED MODULE IMPORTS
# ==========================================
from papers.p12_lora_hu.model import LoRALinear
from papers.p14_instructgpt_ouyang.model import RewardModel, PairwiseRankingLoss, compute_kl_penalized_reward
from papers.p17_rag_lewis.model import DensePassageRetriever
from papers.p18_realm_guu.model import MarginalizedMLMLoss
from papers.p22_switch_fedus.model import SwitchMoELayer
from papers.p23_flashattention_dao.model import TiledAttention
from papers.p42_ppo_schulman.model import PPOClippedLoss

# ==========================================
# THE ADVANCED TESTS
# ==========================================

def test_lora_compression_and_flow():
    """Verifies LoRA parameter freezing, rank decomposition, and gradient flow."""
    IN_FEATURES, OUT_FEATURES, RANK, BATCH, SEQ = 128, 128, 8, 4, 32
    x = torch.randn(BATCH, SEQ, IN_FEATURES)
    
    standard_layer = nn.Linear(IN_FEATURES, OUT_FEATURES)
    lora_layer = LoRALinear(standard_layer, r=RANK)
    
    # Initialization and Parameter Checks
    assert not lora_layer.original_layer.weight.requires_grad
    trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    assert trainable_params == (IN_FEATURES * RANK) + (RANK * OUT_FEATURES)
    
    # Forward and Backward Flow
    out = lora_layer(x)
    assert out.shape == (BATCH, SEQ, OUT_FEATURES)
    out.sum().backward()
    assert lora_layer.lora_A.weight.grad is not None
    assert lora_layer.lora_B.weight.grad is not None

def test_instructgpt_alignment_objectives():
    """Verifies Reward Modeling and RLHF constraints."""
    class DummyBackbone(nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], x.shape[1], 128)
            
    # Reward Model
    rm = RewardModel(DummyBackbone(), hidden_size=128)
    scores = rm(torch.randint(0, 100, (4, 10)))
    assert scores.shape == (4, 1)
    
    # Loss Constraints
    loss_fn = PairwiseRankingLoss()
    loss_good = loss_fn(torch.tensor([[5.0]]), torch.tensor([[1.0]]))
    loss_bad = loss_fn(torch.tensor([[1.0]]), torch.tensor([[5.0]]))
    assert loss_bad > loss_good
    
    # KL Penalty
    penalized = compute_kl_penalized_reward(torch.tensor([10.0]), torch.tensor([0.0]), torch.tensor([-4.6]), beta=0.5)
    assert penalized < 10.0

def test_rag_retrieval_dimensions():
    """Verifies Dense Passage Retriever dual-encoder scaling."""
    BATCH, NUM_DOCS, HIDDEN, TOP_K = 4, 100, 128, 3
    retriever = DensePassageRetriever(hidden_size=HIDDEN)
    
    q_emb = torch.randn(BATCH, HIDDEN)
    d_emb = torch.randn(NUM_DOCS, HIDDEN)
    
    scores = retriever(q_emb, d_emb)
    assert scores.shape == (BATCH, NUM_DOCS)
    
    _, top_k_indices = torch.topk(scores, TOP_K, dim=1)
    assert top_k_indices.shape == (BATCH, TOP_K)

def test_realm_marginalized_loss():
    """Verifies End-to-end differentiable retrieval autograd."""
    BATCH, TOP_K, VOCAB = 2, 3, 100
    loss_fn = MarginalizedMLMLoss()
    
    retriever_logits = torch.randn(BATCH, TOP_K, requires_grad=True)
    generator_logits = torch.randn(BATCH, TOP_K, VOCAB, requires_grad=True)
    target_tokens = torch.tensor([42, 88])
    
    loss = loss_fn(retriever_logits, generator_logits, target_tokens)
    loss.backward()
    
    assert retriever_logits.grad is not None
    assert generator_logits.grad is not None

def test_switch_moe_routing():
    """Verifies Sparse MoE tensor geometry and router gradients."""
    moe_layer = SwitchMoELayer(hidden_size=64, intermediate_size=128, num_experts=4)
    x = torch.randn(2, 10, 64)
    
    out = moe_layer(x)
    assert out.shape == x.shape
    
    out.sum().backward()
    assert moe_layer.router.weight.grad is not None

def test_flashattention_tiling_equivalence():
    """Verifies Online Softmax matches O(N^2) memory-heavy attention."""
    BATCH, SEQ, DIM, BLOCK = 2, 256, 32, 64
    q, k, v = torch.randn(BATCH, SEQ, DIM), torch.randn(BATCH, SEQ, DIM), torch.randn(BATCH, SEQ, DIM)
    
    # Naive
    naive_out = F.softmax((q @ k.transpose(-2, -1)) / (DIM ** 0.5), dim=-1) @ v
    
    # Tiled
    tiled_attn = TiledAttention()
    tiled_out = tiled_attn(q, k, v, block_size=BLOCK)
    
    assert torch.abs(naive_out - tiled_out).max().item() < 1e-4

def test_ppo_clipped_objective():
    """Verifies PPO policy update bounding constraints."""
    ppo_loss = PPOClippedLoss()
    adv = torch.tensor([1.0, 1.0, -1.0])
    log_old = torch.tensor([-1.0, -1.0, -1.0])
    log_unsafe = torch.tensor([-0.1, -0.1, -0.1])
    
    loss_unsafe = ppo_loss(log_unsafe, log_old, adv, epsilon=0.2)
    assert loss_unsafe.item() > -1.5