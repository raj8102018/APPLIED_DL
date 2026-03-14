import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from attention import MultiHeadAttention

class EncoderBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.multiheadattention = MultiHeadAttention(d_model, num_heads)
        self.linear_one = nn.Linear(d_model, d_ff)                 
        self.linear_two = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(d_model)                                 
        self.layernorm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attention_output = self.multiheadattention(x, x, x, mask)                   
        output_with_dropout = self.dropout(attention_output)
        layer_norm_output = self.layernorm(x + output_with_dropout)                 

        feed_forward_output = self.linear_two(F.gelu(self.linear_one(layer_norm_output)))   
        ff_dropout = self.dropout(feed_forward_output)                                      
        layer_norm2_output = self.layernorm2(layer_norm_output + ff_dropout)                

        return layer_norm2_output
