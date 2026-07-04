import streamlit as st
import pandas as pd
import os

from utils.data_loader import load_data
from utils.preprocessing import (
    align_synthetic_targets_to_reference,
    preprocess_data,
    get_categorical_columns
)

from models.ctgan_model import CTGANGenerator

st.title("🧬 CTGAN Synthetic Data Generation")

MODEL_PATH = "outputs/models/ctgan_model.pkl"
DATA_PATH = "outputs/generated_data/synthetic_ctgan.csv"

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------
# Model Status
# -------------------------

if os.path.exists(MODEL_PATH):

    st.success(
        "✅ CTGAN Model Already Trained"
    )

else:

    st.warning(
        "❌ CTGAN Model Not Found"
    )

# -------------------------
# Training Parameters
# -------------------------

epochs = st.slider(
    "Epochs",
    5,
    300,
    10
)

num_samples = st.slider(
    "Synthetic Samples",
    100,
    10000,
    500
)

# -------------------------
# Train CTGAN
# -------------------------

if st.button("🚀 Train CTGAN"):

    categorical_columns = get_categorical_columns(df)

    processed_df, _, _ = preprocess_data(df)

    ctgan = CTGANGenerator()

    # overwrite epochs
    ctgan.model._epochs = epochs

    with st.spinner(
        "Training CTGAN..."
    ):

        ctgan.train(
            processed_df,
            []
        )

    os.makedirs(
        "outputs/models",
        exist_ok=True
    )

    ctgan.save(
        MODEL_PATH
    )

    st.success(
        "✅ CTGAN Trained and Saved"
    )

# -------------------------
# Load Existing Model
# -------------------------

if st.button("📂 Load Saved CTGAN"):

    if not os.path.exists(MODEL_PATH):

        st.error(
            "No saved model found."
        )

    else:

        st.session_state[
            "ctgan_loaded"
        ] = True

        st.success(
            "✅ CTGAN Loaded"
        )

# -------------------------
# Generate Synthetic Data
# -------------------------

if st.button(
    "🧪 Generate Synthetic Data"
):

    if not os.path.exists(
        MODEL_PATH
    ):

        st.error(
            "Train CTGAN first."
        )

    else:

        with st.spinner(
            "Generating..."
        ):

            ctgan = CTGANGenerator()

            ctgan.load(
                MODEL_PATH
            )

            synthetic_data = ctgan.generate(
                num_samples
            )

            synthetic_data = align_synthetic_targets_to_reference(
                synthetic_data,
                df
            )

            os.makedirs(
                "outputs/generated_data",
                exist_ok=True
            )

            synthetic_data.to_csv(
                DATA_PATH,
                index=False
            )

        st.success(
            "✅ Synthetic Data Generated"
        )

        st.dataframe(
            synthetic_data.head()
        )

        st.write(
            "Shape:",
            synthetic_data.shape
        )

# -------------------------
# View Existing Dataset
# -------------------------

if os.path.exists(DATA_PATH):

    st.subheader(
        "Previously Generated Data"
    )

    synthetic_df = pd.read_csv(
        DATA_PATH
    )

    st.dataframe(
        synthetic_df.head()
    )

    st.write(
        "Rows:",
        synthetic_df.shape[0]
    )

    st.write(
        "Columns:",
        synthetic_df.shape[1]
    )
