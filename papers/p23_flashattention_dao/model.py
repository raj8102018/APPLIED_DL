import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TiledAttention(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        pass


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int = 128) -> torch.Tensor:

        B, N, H = q.shape

        out = torch.zeros_like(q)

        m = torch.full((B, N, 1), float('-inf'))

        l = torch.zeros((B, N, 1))

        for i in range(0, N, block_size):

            k_block = k[:, i:i+block_size, :]

            v_block = v[:, i:i+block_size, :]

            for j in range(0, N, block_size):

                q_block = q[:, j:j+block_size, :]

                s_block = q_block @ k_block.transpose(-2,-1)

                s_block_scaled = s_block/(math.sqrt(H))

                m_block = m[:, j : j+block_size, :]
                l_block = l[:, j : j+block_size, :]

                local_max = s_block_scaled.max(-1)[0].unsqueeze(2)

                m_new = torch.maximum(m_block, local_max)

                p_block = torch.exp(s_block_scaled - m_new) 

                l_new = l_block * torch.exp(m_block - m_new) + p_block.sum(dim=-1, keepdim=True)

                out_block = out[:, j:j+block_size, :]

                out_new = out_block * torch.exp(m_block - m_new) + p_block @ v_block

                m[:, j : j+block_size, :] = m_new

                l[:, j : j+block_size, :] = l_new

                out[:, j:j+block_size, :] = out_new
        
        return out/l