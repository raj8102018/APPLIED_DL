import torch
import torch.nn as nn


class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.gamma = nn.Parameter(torch.ones(self.normalized_shape))
        self.beta = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.var(x, unbiased=False, dim=-1, keepdim=True)
        norm = (x - mean)/torch.sqrt(variance+self.eps)
        final_value = norm*self.gamma + self.beta
        return final_value



