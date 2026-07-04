import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd

from utils.data_loader import load_data
from utils.preprocessing import (
    align_synthetic_targets_to_reference,
    preprocess_data
)

from models.vae import VAE


st.title(
    "🧬 VAE Synthetic Data Generation"
)

df = load_data()

processed_df, _, _ = preprocess_data(
    df
)

input_dim = processed_df.shape[1]

epochs = st.slider(
    "Epochs",
    5,
    100,
    10
)

latent_dim = st.slider(
    "Latent Dimension",
    8,
    64,
    16
)

if st.button(
    "Train VAE"
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    data_tensor = torch.tensor(
        processed_df.values,
        dtype=torch.float32
    ).to(device)

    vae = VAE(
        input_dim=input_dim,
        latent_dim=latent_dim
    ).to(device)

    optimizer = torch.optim.Adam(
        vae.parameters(),
        lr=0.001
    )

    progress_bar = st.progress(0)

    loss_container = st.empty()

    for epoch in range(
        epochs
    ):

        vae.train()

        reconstruction, mu, log_var = vae(
            data_tensor
        )

        recon_loss = F.mse_loss(
            reconstruction,
            data_tensor
        )

        kl_loss = -0.5 * torch.mean(

            1 +
            log_var -
            mu.pow(2) -
            log_var.exp()

        )

        loss = (
            recon_loss +
            kl_loss
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        progress_bar.progress(
            (epoch + 1) / epochs
        )

        loss_container.write(
            f"Epoch {epoch+1}/{epochs} | Loss = {loss.item():.4f}"
        )

   import os

os.makedirs("outputs/models", exist_ok=True)

torch.save(
    model.state_dict(),
    "outputs/models/vae_model.pth"
)

    st.success(
        "VAE Training Completed"
    )

    vae.eval()

    with torch.no_grad():

        z = torch.randn(
            500,
            latent_dim
        ).to(device)

        synthetic_data = vae.decode(
            z
        )

        synthetic_df = pd.DataFrame(

            synthetic_data.cpu().numpy(),

            columns=processed_df.columns

        )

        synthetic_df = align_synthetic_targets_to_reference(
            synthetic_df,
            df
        )

        synthetic_df.to_csv(

            "outputs/generated_data/synthetic_vae.csv",

            index=False

        )

    st.success(
        "Synthetic VAE Dataset Generated"
    )

    st.write(
        synthetic_df.head()
    )

    st.write(
        synthetic_df.shape
    )

