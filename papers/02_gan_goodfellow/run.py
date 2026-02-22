import torch
from model import Generator, Discriminator

# Hyperparameters
BATCH_SIZE = 64
LATENT_DIM = 100
IMG_DIM = 784 # 28x28 flattened

# Init
gen = Generator(LATENT_DIM, IMG_DIM)
disc = Discriminator(IMG_DIM)

# Dummy inputs
z = torch.randn(BATCH_SIZE, LATENT_DIM) # Latent noise
real_img = torch.randn(BATCH_SIZE, IMG_DIM) # Simulated real image

# Forward passes
fake_img = gen(z)
pred_real = disc(real_img)
pred_fake = disc(fake_img)

# Assertions
assert fake_img.shape == (BATCH_SIZE, IMG_DIM), "Generator output shape mismatch!"
assert pred_real.shape == (BATCH_SIZE, 1), "Discriminator output shape mismatch!"
assert torch.all((pred_real >= 0) & (pred_real <= 1)), "Discriminator must output probabilities [0, 1]!"
assert torch.all((fake_img >= -1) & (fake_img <= 1)), "Generator output must be bounded (Tanh/Sigmoid)!"

print("SUCCESS: GAN Architectures and Tensor bounds are perfectly aligned.")