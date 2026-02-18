import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
    
    def forward (self, Q, K, V, mask=None):
        #input [batch, number_of_heads, sequence_length, d_model] for all Q,K,V
        d_k = Q.size(-1)                                                          #fetching the dimension since d_k = d_model
        K_t = K.transpose(-1,-2)                                                  #transposing last two dimensions of the keys matrix for multiplication
        scaled_value = torch.matmul(Q,K_t)/math.sqrt(d_k)                    #scaling the dot product
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            scaled_value=  scaled_value.masked_fill(~mask,-1e9)         #masking
        final_value = torch.softmax(scaled_value,-1)@V                  #softmax over last dimension because we are normalizing it
        return final_value

class MultiHeadAttention(nn.Module):
    
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.n = num_heads                              #Initialize number of heads and model dimension
        self.D = d_model
        self.W_q = nn.Linear(d_model,d_model)               #initialize weights, vaswani et al paper states that the queries, keys and values are to be linearly projected h times with different learned linear projections
        self.W_k = nn.Linear(d_model,d_model)               #MultiHead(Q,K,V) = Concat(h_1,h_2,....h_n)*W_o W_o is the output projection
        self.W_v = nn.Linear(d_model,d_model)               #head_i = Attention(QW_qi,KW_ki,V_Wvi)
        self.W_o = nn.Linear(d_model,d_model)
        self.attention = ScaledDotProductAttention(num_heads)

    
    def split_heads(self,X):
        #Assuming the input format for Q,K,V to be [batch, sequence_len, d_model]
        B,S,D = X.shape
        n = self.n
        X = X.reshape(B,S,n,D//n)
        return X
    
    def merge_heads(self,X):
        # Merging the heads after attention
        B, n , S, d = X.shape
        n = self.n
        X = X.transpose(-2,-3)
        X = X.reshape(B,S,d*n) # you could use .view instead by making the tensor contiguous using .contiguous() method
        return X

    def forward  (self,Q,K,V,mask):
        
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        Q = self.split_heads(Q).transpose(-2,-3)             #Using self.split_heads from the same parent class to swap number of heads and sequence length
        K = self.split_heads(K).transpose(-2,-3)
        V = self.split_heads(V).transpose(-2,-3)

        attended_values = self.attention(Q,K,V,mask)
        
        attended_values = self.merge_heads(attended_values)  #Using self.merge_heads to merge all the heads i.e concatenate the values before running through output projection
        output_values = self.W_o(attended_values)
        return output_values


        
