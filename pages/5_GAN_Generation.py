import streamlit as st
import torch
import torch.nn as nn
import pandas as pd

from utils.data_loader import load_data
from utils.preprocessing import (
    align_synthetic_targets_to_reference,
    preprocess_data
)

from models.gan import (
    Generator,
    Discriminator
)

st.title(
    "⚔️ GAN Synthetic Data Generation"
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

noise_dim = st.slider(
    "Noise Dimension",
    10,
    200,
    100
)

if st.button(
    "Train GAN"
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    real_data = torch.tensor(
        processed_df.values,
        dtype=torch.float32
    ).to(device)

    generator = Generator(
        noise_dim,
        input_dim
    ).to(device)

    discriminator = Discriminator(
        input_dim
    ).to(device)

    criterion = nn.BCELoss()

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=0.0002
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=0.0002
    )

    progress_bar = st.progress(0)

    status = st.empty()

    batch_size = 128

    for epoch in range(epochs):

        permutation = torch.randperm(
            real_data.size(0)
        )

        for i in range(
            0,
            real_data.size(0),
            batch_size
        ):

            indices = permutation[
                i:i+batch_size
            ]

            real_batch = real_data[
                indices
            ]

            current_batch = real_batch.size(0)

            real_labels = torch.ones(
                current_batch,
                1
            ).to(device)

            fake_labels = torch.zeros(
                current_batch,
                1
            ).to(device)

            # --------------------
            # Train Discriminator
            # --------------------

            optimizer_d.zero_grad()

            real_output = discriminator(
                real_batch
            )

            real_loss = criterion(
                real_output,
                real_labels
            )

            noise = torch.randn(
                current_batch,
                noise_dim
            ).to(device)

            fake_data = generator(
                noise
            )

            fake_output = discriminator(
                fake_data.detach()
            )

            fake_loss = criterion(
                fake_output,
                fake_labels
            )

            d_loss = (
                real_loss +
                fake_loss
            )

            d_loss.backward()

            optimizer_d.step()

            # --------------------
            # Train Generator
            # --------------------

            optimizer_g.zero_grad()

            noise = torch.randn(
                current_batch,
                noise_dim
            ).to(device)

            generated_data = generator(
                noise
            )

            output = discriminator(
                generated_data
            )

            g_loss = criterion(
                output,
                real_labels
            )

            g_loss.backward()

            optimizer_g.step()

        progress_bar.progress(
            (epoch + 1) / epochs
        )

        status.write(
            f"Epoch {epoch+1}/{epochs} | "
            f"D Loss: {d_loss.item():.4f} | "
            f"G Loss: {g_loss.item():.4f}"
        )

    torch.save(

        generator.state_dict(),

        "outputs/models/gan_generator.pth"

    )

    torch.save(

        discriminator.state_dict(),

        "outputs/models/gan_discriminator.pth"

    )

    generator.eval()

    with torch.no_grad():

        noise = torch.randn(
            500,
            noise_dim
        ).to(device)

        synthetic_data = generator(
            noise
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

            "outputs/generated_data/synthetic_gan.csv",

            index=False

        )

    st.success(
        "GAN Training Completed"
    )

    st.success(
        "Synthetic GAN Dataset Generated"
    )

    st.write(
        synthetic_df.head()
    )

    st.write(
        synthetic_df.shape
    )
