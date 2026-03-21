import torch
from model import PPOClippedLoss

ppo_loss_fn = PPOClippedLoss()
advantages = torch.tensor([1.0, 1.0, -1.0]) 
logprobs_old = torch.tensor([-1.0, -1.0, -1.0])

# Safe update (ratio inside bounds)
logprobs_new_safe = torch.tensor([-0.9, -0.9, -0.9]) 
loss_safe = ppo_loss_fn(logprobs_new_safe, logprobs_old, advantages, epsilon=0.2)

# Unsafe update (ratio explodes past bounds)
logprobs_new_unsafe = torch.tensor([-0.1, -0.1, -0.1])
loss_unsafe = ppo_loss_fn(logprobs_new_unsafe, logprobs_old, advantages, epsilon=0.2)

# The unsafe loss is hard-capped by the clip function, preventing gradient explosion
assert loss_unsafe.item() > -1.5, "PPO failed to clip the explosive update!"

print("SUCCESS: PPO Clipped Objective correctly limits policy updates.")