import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        
        d_k = Q.size(-1)                                                
        K_t = K.transpose(-1,-2)                                        
        scaled_value = torch.matmul(Q,K_t)/math.sqrt(d_k)               
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            scaled_value = scaled_value.masked_fill(~mask,-1e9)         
        final_value = torch.softmax(scaled_value,-1) @ V                  
        return final_value
