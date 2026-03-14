import torch
import torch.nn as nn
from layernorm import CustomLayerNorm
from encoder import EncoderBlock

class BERTEmbeddings(nn.Module):

    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int, type_vocab_size: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size,hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings,hidden_size)
        self.segment_embedding = nn.Embedding(type_vocab_size,hidden_size)
        self.customlayernorm = CustomLayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=0.1)        

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor):
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0).to(input_ids.device)
        input_representation = self.token_embedding(input_ids)+self.position_embedding(position_ids)+self.segment_embedding(segment_ids)
        layernorm_out = self.customlayernorm(input_representation)
        dropout_value = self.dropout(layernorm_out)

        return dropout_value

class BERT(nn.Module):

    def __init__(self, N: int, vocab_size: int, hidden_size: int, max_position_embeddings: int, type_vocab_size: int, d_ff: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embeddings = BERTEmbeddings(vocab_size, hidden_size, max_position_embeddings, type_vocab_size)
        self.encoder_stack = nn.ModuleList([EncoderBlock(hidden_size, num_heads, d_ff, dropout) for _ in range(N)]) ## we can't run a single encoder block in for loop N times i.e 6 times, this results in using same block 6 times and also weights are shared across layers
        self.pooler = nn.Linear(hidden_size,hidden_size)
        self.pooler_act = nn.Tanh()

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor, attention_mask: torch.Tensor):
        embeddings_output = self.embeddings(input_ids, segment_ids)
        hidden_states = embeddings_output
        for layer in self.encoder_stack:
            hidden_states = layer(hidden_states, attention_mask)
        sequence_output = hidden_states
        pooler_output = self.pooler(hidden_states[:, 0, :])
        pooler_finalout = self.pooler_act(pooler_output)

        return sequence_output, pooler_finalout

class BERTPreTrainingHeads(nn.Module):
    
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.GELU(),
            CustomLayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size)
        )
        self.nsp_head = nn.Linear(hidden_size,2)
    def forward(self, sequence_output, pooled_output):

        mlm_head_output = self.mlm_head(sequence_output)
        nsp_head_output = self.nsp_head(pooled_output)

        return mlm_head_output, nsp_head_output