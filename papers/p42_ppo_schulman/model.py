import torch
import torch.nn as nn

class PPOClippedLoss(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        pass
    
    def forward(self, logprobs_new: torch.Tensor, logprobs_old: torch.Tensor, advantages: torch.Tensor, epsilon: float = 0.2) -> torch.Tensor:

        ratio = torch.exp(logprobs_new - logprobs_old)

        unclipped = ratio * advantages

        clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

        clipped = clipped_ratio * advantages

        surrogate = torch.min(unclipped, clipped)

        return -surrogate.mean()