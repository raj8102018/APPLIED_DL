import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6) -> None:

        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        variance = x.pow(2).mean(-1, keepdim=True)

        scaled_output = x * torch.rsqrt(variance + self.eps)

        return scaled_output * self.weight

class LlamaMLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:

        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gate_val = self.gate_proj(x)

        silu_act = F.silu(gate_val)

        up_proj_val = self.up_proj(x)

        gate_out = silu_act * up_proj_val

        final_val = self.down_proj(gate_out)

        return final_val

def rotate_half(x: torch.Tensor) -> torch.Tensor:

    x1, x2 = x.chunk(2, dim=-1)
    
    return torch.cat((-x2, x1), dim=-1)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        
        freqs = torch.outer(t, inv_freq)
        
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        
        cos = self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype, device=x.device)
        return cos, sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed