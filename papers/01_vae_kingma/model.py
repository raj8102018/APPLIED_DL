import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):                   #writing encoder class that takes input and gives two variables mu and log_var as output. log var instead of var for var to be popsitive (even if nn can output any number). we exponentiate the output later to get a valid standard deviation which is positive
    def __init__(self, image_dim, latent_dim):
        super().__init__()
        self.hidden_dim = 400
        self.layer1 = nn.Linear(image_dim, self.hidden_dim)
        self.layer1_act = nn.ReLU()
        self.linear_mu = nn.Linear(self.hidden_dim, latent_dim)
        self.linear_var = nn.Linear(self.hidden_dim, latent_dim)

    def forward(self, x):
        layer1_output = self.layer1(x)
        hidden_layer_output = self.layer1_act(layer1_output)
        mu = self.linear_mu(hidden_layer_output)
        log_var = self.linear_var(hidden_layer_output)

        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, image_dim, latent_dim):
        super().__init__()
        self.hidden_dim = 400
        self.layer1 = nn.Linear(latent_dim, self.hidden_dim)
        self.layer1_act = nn.ReLU()
        self.linear_out = nn.Linear(self.hidden_dim, image_dim)
        self.output_act = nn.Sigmoid()
    
    def forward(self,z):
        layer1_output = self.layer1(z)
        hidden_layer_output = self.layer1_act(layer1_output)
        decoder_out = self.linear_out(hidden_layer_output)
        decoder_out_act = self.output_act(decoder_out)

        return decoder_out_act

class VAE(nn.Module):
    def __init__(self, image_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(image_dim,latent_dim)
        self.decoder = Decoder(image_dim,latent_dim)
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        noise = torch.randn_like(std)
        return mu + std * noise

    def forward(self,x):
        enc_out = self.encoder(x)
        mu, log_var = enc_out
        z = self.reparameterize(mu, log_var)
        dec_out = self.decoder(z)
        
        return dec_out, mu, log_var

def vae_loss_function(recon_x, x, mu, log_var):
    reconstruct_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KL_term = - 0.5 * torch.sum(1+log_var - (torch.square(mu)) - torch.exp(log_var))

    return reconstruct_loss + KL_term
     

