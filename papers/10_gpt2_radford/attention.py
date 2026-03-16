import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int) -> None:
        super().__init__()
        self.d_k = d_k
    
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


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.n = num_heads                              
        self.D = d_model
        self.W_q = nn.Linear(d_model,d_model)               
        self.W_k = nn.Linear(d_model,d_model)               
        self.W_v = nn.Linear(d_model,d_model)               
        self.W_o = nn.Linear(d_model,d_model)
        self.attention = ScaledDotProductAttention(num_heads)

    
    def split_heads(self, X: torch.Tensor) -> torch.Tensor:
        
        B,S,D = X.shape
        n = self.n
        X = X.reshape(B,S,n,D//n)
        return X
    
    def merge_heads(self, X: torch.Tensor) -> torch.Tensor:
        
        B, n , S, d = X.shape
        n = self.n
        X = X.transpose(-2,-3)
        X = X.reshape(B,S,d*n) 
        return X

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        Q = self.split_heads(Q).transpose(-2,-3)            
        K = self.split_heads(K).transpose(-2,-3)
        V = self.split_heads(V).transpose(-2,-3)

        attended_values = self.attention(Q,K,V,mask)
        
        attended_values = self.merge_heads(attended_values)  
        output_values = self.W_o(attended_values)
        return output_values