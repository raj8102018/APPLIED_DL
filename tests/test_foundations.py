import torch
import pytest

# ==========================================
# STANDARD IMPORTS (Using exact folder names from your catalog)
# ==========================================
from papers.p01_vae_kingma.model import VAE, vae_loss_function
from papers.p02_gan_goodfellow.model import Generator, Discriminator
from papers.p04_adam_kingma.optimizer import CustomAdam
from papers.p06_resnet_he.model import ResidualBlock
from papers.p07_alexnet_krizhevsky.model import CustomAlexNet

# ==========================================
# TEST HYPERPARAMETERS
# ==========================================
BATCH_SIZE = 8
IMG_DIM = 784     # For VAE/GAN flattened images
LATENT_DIM = 20
IMG_CHANNELS = 3  # For CNNs
IMG_SIZE = 64     # For CNNs

# ==========================================
# THE FOUNDATIONS TESTS
# ==========================================

def test_vae_forward_backward():
    """Verifies VAE Architecture, ELBO Loss, and Gradient Flow."""
    model = VAE(IMG_DIM, LATENT_DIM)
    dummy_images = torch.rand(BATCH_SIZE, IMG_DIM) 
    
    # Forward Pass
    recon_images, mu, log_var = model(dummy_images)
    loss = vae_loss_function(recon_images, dummy_images, mu, log_var)
    
    # Mathematical Assertions
    assert recon_images.shape == (BATCH_SIZE, IMG_DIM)
    assert mu.shape == (BATCH_SIZE, LATENT_DIM)
    assert loss.item() > 0
    
    # Autograd Check
    loss.backward()
    assert next(model.parameters()).grad is not None, "VAE gradient flow failed"

def test_gan_forward_backward():
    """Verifies GAN Architectures, Tensor bounds, and Gradient Flow."""
    gen = Generator(LATENT_DIM, IMG_DIM)
    disc = Discriminator(IMG_DIM)
    
    z = torch.randn(BATCH_SIZE, LATENT_DIM) 
    real_img = torch.randn(BATCH_SIZE, IMG_DIM) 
    
    # Forward Passes
    fake_img = gen(z)
    pred_real = disc(real_img)
    pred_fake = disc(fake_img.detach()) # Detach for disc training simulation
    
    # Mathematical Assertions
    assert fake_img.shape == (BATCH_SIZE, IMG_DIM)
    assert pred_real.shape == (BATCH_SIZE, 1)
    assert torch.all((pred_real >= 0) & (pred_real <= 1)), "Discriminator bounds failed"
    assert torch.all((fake_img >= -1) & (fake_img <= 1)), "Generator bounds failed"
    
    # Autograd Checks
    pred_real.sum().backward()
    assert next(disc.parameters()).grad is not None, "Discriminator gradient flow failed"
    
    fake_img_train = gen(z)
    pred_fake_train = disc(fake_img_train)
    pred_fake_train.sum().backward()
    assert next(gen.parameters()).grad is not None, "Generator gradient flow failed"

def test_adam_optimizer_parity():
    """Verifies Custom Adam perfectly matches PyTorch official C++ backend."""
    x_custom = torch.tensor([10.0], requires_grad=True)
    x_official = torch.tensor([10.0], requires_grad=True)
    
    opt_custom = CustomAdam([x_custom], lr=0.1)
    opt_official = torch.optim.Adam([x_official], lr=0.1)
    
    for _ in range(50):
        # Custom pass
        opt_custom.zero_grad()
        (x_custom ** 2).backward()
        opt_custom.step()
        
        # Official pass
        opt_official.zero_grad()
        (x_official ** 2).backward()
        opt_official.step()
        
    diff = torch.abs(x_custom - x_official).item()
    assert diff < 1e-5, f"Optimizer parity failed. Diff: {diff}"

def test_resnet_block_forward_backward():
    """Verifies ResNet Identity/Projection logic and Skip Connection gradients."""
    x = torch.randn(16, 64, 32, 32, requires_grad=True)
    
    # Identity
    identity_block = ResidualBlock(in_channels=64, out_channels=64, stride=1)
    out_identity = identity_block(x)
    assert out_identity.shape == (16, 64, 32, 32)
    
    # Projection
    projection_block = ResidualBlock(in_channels=64, out_channels=128, stride=2)
    out_projection = projection_block(x)
    assert out_projection.shape == (16, 128, 16, 16)
    
    # Autograd Check
    out_projection.sum().backward()
    assert x.grad is not None, "Skip connection gradient flow failed"
    assert next(projection_block.parameters()).grad is not None

def test_alexnet_forward_backward():
    """Verifies AlexNet dimensional reduction and Gradient Flow."""
    num_classes = 100
    model = CustomAlexNet(num_classes=num_classes)
    x = torch.randn(BATCH_SIZE, 3, 224, 224) # Standard ImageNet dim
    
    logits = model(x)
    assert logits.shape == (BATCH_SIZE, num_classes)
    
    logits.sum().backward()
    assert next(model.parameters()).grad is not None, "AlexNet gradient flow failed"