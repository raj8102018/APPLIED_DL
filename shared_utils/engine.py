import torch
from tqdm import tqdm

class BaseEngine:
    def __init__(self, device=None):
        """Automatically routes to the best available silicon."""
        self.device = device if device else self._get_device()
        print(f"Engine initialized on computing device: {self.device}")

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

class SupervisedEngine(BaseEngine):
    def __init__(self, model, optimizer, criterion, device=None):
        super().__init__(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def train_epoch(self, dataloader, epoch_idx, step_fn=None):
        """
        step_fn: A custom function for models with weird inputs (like Transformers needing masks).
                 If None, uses standard (input, target) routing.
        """
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx} [Train]", leave=False)

        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Use custom step function if provided (e.g., for Transformer masks), else standard
            if step_fn:
                loss = step_fn(self.model, batch, self.criterion, self.device)
            else:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        return total_loss / len(dataloader)


class AdversarialEngine(BaseEngine):
    def __init__(self, generator, discriminator, opt_gen, opt_disc, criterion, device=None):
        super().__init__(device)
        self.gen = generator.to(self.device)
        self.disc = discriminator.to(self.device)
        self.opt_gen = opt_gen
        self.opt_disc = opt_disc
        self.criterion = criterion

    def train_epoch(self, dataloader, epoch_idx, latent_dim):
        self.gen.train()
        self.disc.train()
        
        total_d_loss = 0.0
        total_g_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx} [GAN]", leave=False)

        for real_data in pbar:
            # Note: Assuming dataloader returns batches of real data
            real_data = real_data[0].to(self.device) if isinstance(real_data, (list, tuple)) else real_data.to(self.device)
            batch_size = real_data.size(0)

            # 1. Train Discriminator
            self.opt_disc.zero_grad()
            z = torch.randn(batch_size, latent_dim, device=self.device)
            fake_data = self.gen(z)
            
            real_loss = self.criterion(self.disc(real_data), torch.ones(batch_size, 1, device=self.device))
            fake_loss = self.criterion(self.disc(fake_data.detach()), torch.zeros(batch_size, 1, device=self.device))
            d_loss = real_loss + fake_loss
            d_loss.backward()
            self.opt_disc.step()

            # 2. Train Generator
            self.opt_gen.zero_grad()
            g_loss = self.criterion(self.disc(fake_data), torch.ones(batch_size, 1, device=self.device))
            g_loss.backward()
            self.opt_gen.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            pbar.set_postfix({'d_loss': f"{d_loss.item():.4f}", 'g_loss': f"{g_loss.item():.4f}"})

        return total_d_loss / len(dataloader), total_g_loss / len(dataloader)