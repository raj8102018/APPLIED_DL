import torch
import torch.nn as nn

class DensePassageRetriever(nn.Module):

    def __init__(self, hidden_size: int) -> None:

        super().__init__()

        self.query_encoder = nn.Linear(hidden_size, hidden_size)

        self.doc_encoder = nn.Linear(hidden_size, hidden_size)

    def forward(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:

        query_pass = self.query_encoder(query_embeddings)
        doc_pass = self.doc_encoder(doc_embeddings)

        scores = torch.matmul( query_pass, torch.transpose(doc_pass,0,1))

        return scores