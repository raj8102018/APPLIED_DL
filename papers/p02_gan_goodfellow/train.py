import torch
import torch.nn as nn
from model import Generator, Discriminator
from tqdm import tqdm


BATCH_SIZE = 64
latent_dim = 100
image_dim = 784


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disc = Discriminator(image_dim).to(device)
    gen = Generator(latent_dim,image_dim).to(device)
    criterion = nn.BCELoss()
    opt_disc = torch.optim.Adam(disc.parameters(), lr=1e-3)
    opt_gen = torch.optim.Adam(gen.parameters(), lr=1e-4)
    for epoch in tqdm(range(1), desc = "Epochs"):
        
        for i in tqdm(range(5),desc = "Batches", leave=False):
            #zero the gradients
            batch_data = torch.randn(BATCH_SIZE,784).to(device)
            opt_disc.zero_grad()
            disc_out = disc(batch_data)
            loss_batch_data = criterion(disc_out, torch.ones(BATCH_SIZE, 1)).to(device)
            z = torch.randn(BATCH_SIZE,100).to(device)
            fake_data = gen(z)
            disc_out2 = disc(fake_data.detach())
            loss_fake_data = criterion(disc_out2, torch.zeros(BATCH_SIZE,1)).to(device)
            total_loss = torch.add(loss_batch_data, loss_fake_data)
            total_loss.backward()
            opt_disc.step()
            
            opt_gen.zero_grad()
            disc_out_3 = disc(fake_data)
            loss_fake_data_2 = criterion(disc_out_3, torch.ones(BATCH_SIZE,1)).to(device)
            loss_fake_data_2.backward()
            opt_gen.step()
    print("training loop ran successfully")


if __name__ == "__main__":
    main()


