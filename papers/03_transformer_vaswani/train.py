import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader, random_split # Added random_split for dataset splitting

# --- Hyperparameters (re-defined here to ensure independence if cells run out of order) ---
VOCAB_SIZE = 100
PAD_ID = 0  # Also used as BOS for shifted decoder input
SEQ_LEN = 20
BATCH_SIZE = 32
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 512
DROPOUT = 0.1
LR = 1e-3 # This LR is different from LOOP 1
EPOCHS = 100

# Re-define CopyDataset and create_masks here to ensure they are available for this cell
# This is a safe practice if cells can be run independently or out of order.
class CopyDataset(Dataset):
    """Dummy seq2seq dataset: target = source (copy task). Random integer sequences."""

    def __init__(self, num_samples, seq_len, vocab_size, pad_id=0, seed=None):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        if seed is not None:
            torch.manual_seed(seed)
        # Random ints 1..vocab_size-1 to avoid colliding with pad_id
        self.data = torch.randint(1, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        src = self.data[idx].clone()
        # Decoder input: shifted right with PAD_ID at start
        tgt_input = torch.empty_like(src)
        tgt_input[0] = self.pad_id
        tgt_input[1:] = src[:-1]
        return src, tgt_input  # labels, decoder input


def create_masks(batch_size, seq_len, device):
    """Create src_mask (all ones, no padding) and tgt_mask (causal)."""
    src_mask = torch.ones(batch_size, 1, 1, seq_len, device=device)
    tgt_mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, device=device))
    return src_mask, tgt_mask

# Assuming Transformer class is defined in an earlier cell and accessible
# If not, it would need to be included here as well.

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & DataLoader
    full_dataset = CopyDataset(
        num_samples=10_000, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, pad_id=PAD_ID, seed=42 # Changed num_samples to 10_000
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0 # Added validation loader
    )

    # Model
    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        N=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
        max_len=SEQ_LEN,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_losses = []  # Initialize list to store train losses
    val_losses = []    # Initialize list to store validation losses

    for epoch in range(EPOCHS):
        # --- Training Loop ---
        model.train() # Set model to training mode
        total_loss = 0.0
        num_batches = 0
        for src, tgt_input in train_loader:
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            src_mask, tgt_mask = create_masks(src.size(0), src.size(1), device)

            optimizer.zero_grad()
            logits = model(src, tgt_input, src_mask, tgt_mask)  # [B, T, vocab_size]

            # CrossEntropyLoss: predict src at each position
            logits_flat = logits.view(-1, VOCAB_SIZE)
            labels_flat = src.contiguous().view(-1)

            loss = criterion(logits_flat, labels_flat)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss) # Append average loss for the epoch

        # --- Validation Loop ---
        model.eval() # Set model to evaluation mode
        total_val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad(): # Disable gradient calculations
            for src_val, tgt_input_val in val_loader:
                src_val = src_val.to(device)
                tgt_input_val = tgt_input_val.to(device)
                src_mask_val, tgt_mask_val = create_masks(src_val.size(0), src_val.size(1), device)

                logits_val = model(src_val, tgt_input_val, src_mask_val, tgt_mask_val)

                logits_val_flat = logits_val.view(-1, VOCAB_SIZE)
                labels_val_flat = src_val.contiguous().view(-1)

                loss_val = criterion(logits_val_flat, labels_val_flat)

                total_val_loss += loss_val.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches
        val_losses.append(avg_val_loss)

        model.train() # Set model back to training mode

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    print("Training complete.")

    global global_train_losses_2 # Make it accessible outside main
    global_train_losses_2 = train_losses
    global global_val_losses_2 # Make it accessible outside main
    global_val_losses_2 = val_losses

if __name__ == "__main__":
    main()

# generated code block for plots
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# plt.plot(global_train_losses_2, label='Training Loss')
# plt.plot(global_val_losses_2, label='Validation Loss')
# plt.title('Training and Validation Loss Curve (Run 2: LR = 1e-3)')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()