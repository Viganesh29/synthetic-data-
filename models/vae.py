import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim=16
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU()

        )

        self.mu = nn.Linear(
            64,
            latent_dim
        )

        self.log_var = nn.Linear(
            64,
            latent_dim
        )

        self.decoder = nn.Sequential(

            nn.Linear(
                latent_dim,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                input_dim
            ),

            nn.Sigmoid()

        )

    def encode(self, x):

        x = self.encoder(x)

        return (
            self.mu(x),
            self.log_var(x)
        )

    def reparameterize(
        self,
        mu,
        log_var
    ):

        std = torch.exp(
            0.5 * log_var
        )

        eps = torch.randn_like(
            std
        )

        return mu + eps * std

    def decode(self, z):

        return self.decoder(z)

    def forward(self, x):

        mu, log_var = self.encode(x)

        z = self.reparameterize(
            mu,
            log_var
        )

        reconstruction = self.decode(z)

        return (
            reconstruction,
            mu,
            log_var
        )