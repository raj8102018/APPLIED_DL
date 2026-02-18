import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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

class DecoderBlock_With_Dynamic_Mask(nn.Module):
    
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super().__init__()
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.multiheadattention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.linear_one = nn.Linear(d_model, d_ff)
        self.linear_two = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

    def generate_causal_mask(self,seq_len, device):  #We are generating causal mask here dynamically once for every forward, which is not how it is done it production.
        mask = torch.ones(1,1,seq_len,seq_len, device=device) # this is simple, self-contained but less flexible and less reusable in advanced pipelines
        mask = torch.tril(mask)
        return mask

    def sublayer_1(self,x,mask=None):
        # assuming the Q,K,V input to be [batch_size, seq_len, d_model]
        seq_len = x.shape[1]
        mask = self.generate_causal_mask(seq_len,x.device)
        masked_self_attention = self.multiheadattention(x,x,x,mask)
        output_with_dropout = self.dropout(masked_self_attention)
        layer_norm_output = self.layernorm(x + output_with_dropout)

        return layer_norm_output #We feed this as query to the next sublayer along with encoder output as key

    def sublayer_2(self,sublayer_1_output,enc_output,src_mask):
        cross_attention=self.cross_attention(sublayer_1_output,enc_output,enc_output,src_mask)
        output_with_dropout = self.dropout(cross_attention)
        layer_norm2_output = self.layernorm2(sublayer_1_output + output_with_dropout)

        feed_forward_output = self.linear_two(F.relu(self.linear_one(layer_norm2_output)))
        ff_dropout = self.dropout(feed_forward_output)                                      #You can also write nn.functional.relu instead of F.relu but it is verbose
        layer_norm3_output = self.layernorm3(layer_norm2_output + ff_dropout)

        return layer_norm3_output   

    def forward(self, x, enc_output, src_mask):
        
        sublayer_1_output = self.sublayer_1(x)
        sublayer_2_output = self.sublayer_2(sublayer_1_output,enc_output,src_mask)

        return sublayer_2_output


class DecoderBlock(nn.Module):
    
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super().__init__()
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.multiheadattention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.linear_one = nn.Linear(d_model, d_ff)
        self.linear_two = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

    def sublayer_1(self,x,mask=None):
        # assuming the Q,K,V input to be [batch_size, seq_len, d_model]
        masked_self_attention = self.multiheadattention(x,x,x,mask)
        output_with_dropout = self.dropout(masked_self_attention)
        layer_norm_output = self.layernorm(x + output_with_dropout)

        return layer_norm_output #We feed this as query to the next sublayer along with encoder output as key

    def sublayer_2(self,sublayer_1_output,enc_output,src_mask):
        cross_attention=self.cross_attention(sublayer_1_output,enc_output,enc_output,src_mask)
        output_with_dropout = self.dropout(cross_attention)
        layer_norm2_output = self.layernorm2(sublayer_1_output + output_with_dropout)

        feed_forward_output = self.linear_two(F.relu(self.linear_one(layer_norm2_output)))
        ff_dropout = self.dropout(feed_forward_output)                                      #You can also write nn.functional.relu instead of F.relu but it is verbose
        layer_norm3_output = self.layernorm3(layer_norm2_output + ff_dropout)

        return layer_norm3_output   

    def forward(self, x, enc_output, src_mask, tgt_mask):
        
        sublayer_1_output = self.sublayer_1(x,tgt_mask)
        sublayer_2_output = self.sublayer_2(sublayer_1_output,enc_output,src_mask)

        return sublayer_2_output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len= 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        dimensional_matrix = torch.zeros(max_len,d_model, dtype = torch.float32)                #shape [max_len,d_model] interpretation row = token position in sequence and col = embedding dimenstion. it's like position n -> vector of size d_model
        positions = torch.arange(max_len, dtype = torch.float32).unsqueeze(1)               #Shape would be [max_len,1]
        i = torch.arange(0, d_model, 2, dtype = torch.float32)
        denominator_term = torch.exp(-(i/d_model) * math.log(10000)).unsqueeze(0)                #shape would be [1, d_model]. Note: We are using log here for numerical stability. this operation basically builds a 2D grid and every cell has unique frequency
        angles = positions * denominator_term               #this becomes [max_len, d_model] --> Angle = position * frequency

        dimensional_matrix[:, 0::2] = torch.sin(angles)                #projecting the values to dimensional matrix after encoding
        dimensional_matrix[:, 1::2] = torch.cos(angles)
        
        positional_encoding_matrix = dimensional_matrix.unsqueeze(0)              #adding a batch dimension to make it [1, max_len, d_model], transformers expect input shape like [batch_size, sequence_length, embedding_dim]

        self.register_buffer('pe', positional_encoding_matrix)              #register_buffer tells pyTorch to save this dimensional matrix with the model state and not update it with gradients when an optimizer is involved
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x

class Transformer(nn.Module):
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, N, num_heads, d_ff, dropout, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embedding_layer1 = nn.Embedding(src_vocab_size, d_model)
        self.embedding_layer2 = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model,dropout,max_len)
        self.encoder_stack = nn.ModuleList([EncoderBlock(d_model,num_heads,d_ff,dropout) for _ in range(N)]) ## we can't run a single encoder block in for loop N times i.e 6 times, this results in using same block 6 times and also weights are shared across layers
        self.decoder_stack = nn.ModuleList([DecoderBlock(d_model,num_heads,d_ff,dropout) for _ in range(N)])
        self.output_layer = nn.Linear(d_model,tgt_vocab_size)

    def forward(self,src,tgt,src_mask,tgt_mask):
        embeddings_1 = self.embedding_layer1(src)*math.sqrt(self.d_model)
        hidden_states_1 = self.positional_encoding(embeddings_1)
        for sheet in self.encoder_stack:
            hidden_states_1 = sheet(hidden_states_1,src_mask)  #adding mask here is necessary, in a real batch if we have to pad tokens to fir them into a tensor and we don't pass the mask, the efficiency goes down as self attention mechanism will waste resources work by calculating attention score to these padding tokens
        encoder_output = hidden_states_1
        embeddings_2 = self.embedding_layer2(tgt)*math.sqrt(self.d_model)
        hidden_states_2 = self.positional_encoding(embeddings_2)
        for sheet in self.decoder_stack:
            hidden_states_2 = sheet(hidden_states_2,encoder_output,src_mask,tgt_mask)
        decoder_output = hidden_states_2
        final_output = self.output_layer(decoder_output)

        return final_output
        
        
