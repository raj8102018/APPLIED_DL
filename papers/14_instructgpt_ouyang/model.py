import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):

    def __init__(self, backbone: nn.Module, hidden_size: int) -> None:

        super().__init__()

        self.backbone = backbone
        self.score_head = nn.Linear(hidden_size, 1 , bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:

        hidden_states = self.backbone(input_ids)
        last_hidden_state = hidden_states[:, -1, :]
        score_head_pass = self.score_head(last_hidden_state)

        return score_head_pass
    
    
class PairwiseRankingLoss(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        pass


    def forward(self, chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:

        diff = chosen_rewards - rejected_rewards

        loss = -F.logsigmoid(diff).mean()

        return loss


def compute_kl_penalized_reward(reward: torch.Tensor, logprobs_rl: torch.Tensor, logprobs_sft: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    
    kl_penalty = logprobs_rl - logprobs_sft

    return reward - (beta * kl_penalty)