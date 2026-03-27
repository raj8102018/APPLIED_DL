import torch
import torch.nn as nn

class TiledAttention(nn.Module):

    def __init__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int = 128) -> torch.Tensor:

        super().__init__()

        out = torch.zeros_like(q)