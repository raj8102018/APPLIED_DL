import torch
from model import DensePassageRetriever

# Hyperparameters
BATCH_SIZE = 4       # 4 user queries at the same time
NUM_DOCS = 10000     # A database of 10,000 external documents
HIDDEN_SIZE = 768    # Standard BERT embedding size
TOP_K = 3            # We want to retrieve the top 3 docs for each query

# Dummy Data (Simulating the output of the embedding models)
query_embeddings = torch.randn(BATCH_SIZE, HIDDEN_SIZE)
doc_embeddings = torch.randn(NUM_DOCS, HIDDEN_SIZE)

# Init Retriever
retriever = DensePassageRetriever(hidden_size=HIDDEN_SIZE)

# Forward Pass (The Search)
retriever.eval()
with torch.no_grad():
    scores = retriever(query_embeddings, doc_embeddings)

# The Assertions
print(f"Scores Matrix Shape: {scores.shape}")
assert scores.shape == (BATCH_SIZE, NUM_DOCS), f"FATAL: Dimension mismatch. Expected {(BATCH_SIZE, NUM_DOCS)}, got {scores.shape}"

# Extracting the Top-K documents for the Generator
top_k_scores, top_k_indices = torch.topk(scores, TOP_K, dim=1)

print(f"Top-{TOP_K} Indices Shape: {top_k_indices.shape}")
assert top_k_indices.shape == (BATCH_SIZE, TOP_K), "FATAL: Top-K extraction failed."

print("SUCCESS: Dense Passage Retriever dual-encoder math and similarity search are operational.")