import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_dim):               #shape of input z is [batch-size, latent_dim]
        super().__init__()
        self.hidden_dim = 128
        self.linear1 = nn.Linear(latent_dim, self.hidden_dim)
        self.activation1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.activation2 = nn.LeakyReLU()
        self.linear3 = nn.Linear(self.hidden_dim, img_dim)
        self.activation3 = nn.Tanh()
    
    def forward(self, z):
        linear1_output = self.linear1(z)
        act1_out = self.activation1(linear1_output)
        linear2_out = self.linear2(act1_out)
        act2_out = self.activation2(linear2_out)
        linear3_out = self.linear3(act2_out)
        act3_out = self.activation3(linear3_out)

        return act3_out


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.hidden_dim = 128
        self.dim_output=1
        self.model = nn.Sequential(
                        nn.Linear(img_dim, self.hidden_dim),
                        nn.LeakyReLU(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.LeakyReLU(),
                        nn.Linear(self.hidden_dim, self.dim_output),
                        nn.Sigmoid()
        )
    
    def forward(self,x):
        
        return self.model(x)