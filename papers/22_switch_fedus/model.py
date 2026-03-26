import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertBlock(nn.Module):

    def __init__(self, hidden_size: int, d_ff: int) -> None:

        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, hidden_size, bias=False)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.mlp(x)

class SwitchMoELayer(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int) -> None:

        super().__init__()

        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        self.experts = nn.ModuleList([ExpertBlock(hidden_size, d_ff) for _ in range(N)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        flattend_input = torch.flatten(x, start_dim=0, end_dim=1)

        router_out = self.router(flattend_input)

        router_probs = F.softmax(x, dim=-1)

        routing_weight, expert_id = router_probs.max(dim=-1)

        new_tensor = torch.zeros_like(flattend_input)

        for expert in self.experts:

            



