import torch
import torch.nn as nn
import torch.nn.functional as F

class MarginalizedMLMLoss(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        pass
    
    def forward(self, retriever_logits: torch.Tensor, generator_logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:

        log_p_z = F.log_softmax(retriever_logits, dim=1)

        log_p_y_given_z = F.log_softmax(generator_logits, dim=2)

        B, T, V = generator_logits.shape

        target_tokens_broadcast = target_tokens.unsqueeze(1).unsqueeze(1)

        target_tokens_expanded = target_tokens_broadcast.expand(-1, T, 1)

        target_log_probs = torch.gather(log_p_y_given_z, dim=2, index=target_tokens_expanded)

        target_log_probs_sq = target_log_probs.squeeze(2)

        log_joint = log_p_z + target_log_probs_sq

        marginalized_val = torch.logsumexp(log_joint, dim=1)

        return -marginalized_val.mean()