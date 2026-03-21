import torch
import torch.nn as nn
from model import RewardModel, PairwiseRankingLoss, compute_kl_penalized_reward

# 1. Reward Model Check
class DummyBackbone(nn.Module):
    def forward(self, x):
        return torch.randn(x.shape[0], x.shape[1], 256) # [Batch, Seq, Hidden]

rm = RewardModel(DummyBackbone(), hidden_size=256)
scores = rm(torch.randint(0, 100, (4, 10)))
assert scores.shape == (4, 1), f"Reward shape mismatch: {scores.shape}"

# 2. Ranking Loss Check
loss_fn = PairwiseRankingLoss()
loss_good = loss_fn(torch.tensor([[5.0]]), torch.tensor([[1.0]]))
loss_bad = loss_fn(torch.tensor([[1.0]]), torch.tensor([[5.0]]))
assert loss_bad > loss_good, "Loss must penalize inverted preference."

# 3. KL Check
penalized = compute_kl_penalized_reward(torch.tensor([10.0]), torch.tensor([0.0]), torch.tensor([-4.6]), beta=0.5)
assert penalized < 10.0, "KL penalty failed to reduce reward."

print("SUCCESS: InstructGPT components operational.")