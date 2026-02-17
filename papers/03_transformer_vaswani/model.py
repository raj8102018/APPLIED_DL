import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention

class EncoderBlock(nn.Module):

    def __init__(self,d_model,num_heads,d_ff,dropout):
        super().__init__()
        self.d_model=d_model
        self.d_ff = d_ff
        self.num_heads=num_heads
        self.multiheadattention = MultiHeadAttention(d_model,num_heads)
        self.linear_one = nn.Linear(d_model,d_ff)                  #Transformer FFN structure linear(d_model to d_ff)-->activation -->linear(d_ff to d_model)
        self.linear_two = nn.Linear(d_ff,d_model)
        self.dropout= nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(d_model)                                 #d_ff is typically 4 times d_model
        self.layernorm2 = nn.LayerNorm(d_model)
    
    def forward(self,x,mask=None):
        attention_output = self.multiheadattention(x,x,x,mask)                   #Attention block
        output_with_dropout = self.dropout(attention_output)
        layer_norm_output = self.layernorm(x + output_with_dropout)                 #modern methods use pre-LN i.e before attention happens the residual goes through layernorm i.e x--> layernorm -->attention

        feed_forward_output = self.linear_two(F.relu(self.linear_one(layer_norm_output)))               #FFN block here pre-LN would be x plus output with droput of sublayer
        ff_dropout = self.dropout(feed_forward_output)                                      #You can also write nn.functional.relu instead of F.relu but it is verbose
        layer_norm2_output = self.layernorm2(layer_norm_output + ff_dropout)                #mathematically a single layer norm works but residuals get mixed up

        return layer_norm2_output

class DecoderBlock(nn.Module):
    
    def __init__(self)