import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from attention import ScaledDotProductAttention
from layernorm import CustomLayerNorm


class KVCacheAttention(nn.Module):
    
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size//num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attention = ScaledDotProductAttention()
    
    def forward(self, x: torch.Tensor, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        q_pass = self.q_proj(x)
        k_pass = self.k_proj(x)
        v_pass = self.v_proj(x)
        #B, S, H = q_pass.shape
        #q_pass = q_pass.reshape(B, S, num_heads, head_dim)

        q_pass = q_pass.reshape(*q_pass.shape[:-1], self.num_heads, self.head_dim).transpose(1,2)
        k_pass = k_pass.reshape(*k_pass.shape[:-1], self.num_heads, self.head_dim).transpose(1,2)
        v_pass = v_pass.reshape(*v_pass.shape[:-1], self.num_heads, self.head_dim).transpose(1,2)
        
        if past_kv is not None:
            past_k, past_v = past_kv
            k_pass = torch.cat((past_k,k_pass), dim=2)
            v_pass = torch.cat((past_v,v_pass), dim=2)

        seq_len = x.shape[1]
        extended_len = k_pass.shape[2]
        full_mask = torch.tril(torch.ones(extended_len,extended_len).to(x.device))
        dynamic_mask = full_mask[-seq_len:, :].unsqueeze(0).unsqueeze(0)
        attention_out = self.attention(q_pass, k_pass, v_pass, dynamic_mask).transpose(1,2)
        B, S = attention_out.shape[:2]
        attention_out = attention_out.contiguous().view(B, S, self.hidden_size)
        proj_tensor = self.out_proj(attention_out)

        return proj_tensor, (k_pass, v_pass)


class GPTEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_positions: int = 512) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size,hidden_size)
        self.position_embedding = nn.Embedding(max_positions,hidden_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        position_ids = position_ids
        embeddings = self.token_embedding(input_ids)+self.position_embedding(position_ids)
        final_representation = self.dropout(embeddings)

        return final_representation


class GPTDecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layernorm1 = CustomLayerNorm(hidden_size)
        self.kvcache_attention = KVCacheAttention(hidden_size, num_heads)
        self.layernorm2 = CustomLayerNorm(hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, hidden_size)
        )
        self.dropout=nn.Dropout(p=0.1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] =  None, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        layernorm1_out = self.layernorm1(hidden_states)
        attention_output, present_kv = self.kvcache_attention(layernorm1_out, past_kv=past_kv)
        hidden_states = hidden_states + self.dropout(attention_output)
        layernorm2_out = self.layernorm2(hidden_states)
        feed_forward_output = self.feedforward(layernorm2_out)
        final_output = hidden_states + self.dropout(feed_forward_output)

        return final_output, present_kv


class GPT(nn.Module):

    def __init__(self, vocab_size: int, hidden_size: int, max_positions: int, num_layers: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.max_positions = max_positions
        self.embeddings = GPTEmbeddings(vocab_size, hidden_size, max_positions)
        self.decoder_stack = nn.ModuleList([GPTDecoderBlock(hidden_size, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.layernorm = CustomLayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.token_embedding.weight #weight tying

    
    def forward(self, input_ids: torch.Tensor, past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:

        seq_len = input_ids.shape[1]
        past_seq_len = past_kv[0][0].shape[2] if past_kv is not None else 0
        position_ids = torch.arange(past_seq_len, past_seq_len + seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        embeddings_output = self.embeddings(input_ids, position_ids)
        hidden_states = embeddings_output

        if past_kv is None:
            dynamic_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        else:
            # Shape: [1, 1, 1, past_seq_len + 1] -> allows the 1 new token to see all past tokens + itself
            dynamic_mask = torch.ones(1, 1, 1, past_seq_len + 1, device=input_ids.device)
        presents = []
        for i, layer in enumerate(self.decoder_stack):
            if past_kv is not None:
                hidden_states, present_kv = layer(hidden_states, dynamic_mask, past_kv[i])
            else:
                hidden_states, present_kv = layer(hidden_states, dynamic_mask)
            presents.append(present_kv)

        sequence_output = hidden_states
        layernorm_output = self.layernorm(sequence_output)
        final_output = self.lm_head(layernorm_output)

        return final_output, presents


    def _init_weights(self, module: nn.Module) -> None:

        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = 0.02
            if hasattr(module, "IS_RESIDUAL_PROJECTION") and module.IS_RESIDUAL_PROJECTION:
                std = 0.02/math.sqrt(2 * self.num_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
        
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)



    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:

        past_kv = None

        for _ in range(max_new_tokens):
            max_positions = self.max_positions
            if idx.shape[1] > max_positions:
                idx_stripped = idx[:, -max_positions:]
            else:
                idx_stripped = idx
            if past_kv is None:
                logits, past_kv = self(idx_stripped)
            else:
                idx_stripped = idx[:, -1:]
                logits, past_kv = self(idx_stripped, past_kv=past_kv)
            logits = logits[:, -1, :]
            logits = logits/temperature

            if top_k is not None:
                values, indices = torch.topk(logits, top_k)
                smallest_topk = values[:, -1].unsqueeze(1)
                logits[logits < smallest_topk] = float('-inf')

            outputs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(outputs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)
        
        return idx


