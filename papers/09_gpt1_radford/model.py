import torch
import torch.nn as nn
from layernorm import CustomLayerNorm
from attention import MultiHeadAttention

class GPTEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_positions: int = 512) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size,hidden_size)
        self.position_embedding = nn.Embedding(max_positions,hidden_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0).to(input_ids.device)
        embeddings = self.token_embedding(input_ids)+self.position_embedding(position_ids)
        final_representation = self.dropout(embeddings)

        return final_representation

class GPTDecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layernorm1 = CustomLayerNorm(hidden_size)
        self.multiheadattention = MultiHeadAttention(hidden_size, num_heads)
        self.layernorm2 = CustomLayerNorm(hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, hidden_size)
        )
        self.dropout=nn.Dropout(p=0.1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        layernorm1_out = self.layernorm1(hidden_states)
        attention_output = self.multiheadattention(layernorm1_out,layernorm1_out,layernorm1_out, attention_mask)
        hidden_states = hidden_states + self.dropout(attention_output)
        layernorm2_out = self.layernorm2(hidden_states)
        feed_forward_output = self.feedforward(layernorm2_out)
        final_output = hidden_states + self.dropout(feed_forward_output)

        return final_output

class GPT(nn.Module):

    def __init__(self, vocab_size: int, hidden_size: int, max_positions: int, num_layers: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embeddings = GPTEmbeddings(vocab_size, hidden_size, max_positions)
        self.decoder_stack = nn.ModuleList([GPTDecoderBlock(hidden_size, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.layernorm = CustomLayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.token_embedding.weight #weight tying

    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:

        seq_len = input_ids.shape[1]
        embeddings_output = self.embeddings(input_ids)
        hidden_states = embeddings_output
        dynamic_mask = torch.tril(torch.ones(seq_len,seq_len).to(input_ids.device)).unsqueeze(0).unsqueeze(0)
        for layer in self.decoder_stack:
            hidden_states = layer(hidden_states, dynamic_mask)
        sequence_output = hidden_states
        layernorm_output = self.layernorm(sequence_output)
        final_output = self.lm_head(layernorm_output)

        return final_output
