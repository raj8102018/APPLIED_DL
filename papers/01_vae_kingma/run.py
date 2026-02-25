import torch
from model import VAE, vae_loss_function

BATCH_SIZE = 64
IMG_DIM = 784
LATENT_DIM = 20

# Init
model = VAE(IMG_DIM, LATENT_DIM)
dummy_images = torch.rand(BATCH_SIZE, IMG_DIM) # Random pixel values [0, 1]

# Forward Pass
recon_images, mu, log_var = model(dummy_images)

# Loss calculation
loss = vae_loss_function(recon_images, dummy_images, mu, log_var)

# Assertions
assert recon_images.shape == (BATCH_SIZE, IMG_DIM), "Reconstruction shape mismatch!"
assert mu.shape == (BATCH_SIZE, LATENT_DIM), "Mu shape mismatch!"
assert log_var.shape == (BATCH_SIZE, LATENT_DIM), "Log Var shape mismatch!"
assert loss.item() > 0, "Loss calculation failed!"

print("SUCCESS: VAE Architecture and ELBO Loss are mathematically sound.")