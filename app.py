import streamlit as st

st.set_page_config(
    page_title="Credit Risk Synthetic Data Generation",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Synthetic Financial Data Generation for Credit Risk Assessment")

st.markdown("""
This project investigates the use of:

- Variational Autoencoders (VAE)
- Generative Adversarial Networks (GAN)
- Conditional Tabular GAN (CTGAN)

for:

- Credit Risk Assessment
- Synthetic Data Generation
- Privacy Preservation
- Fairness Analysis

Use the navigation menu on the left to explore the project.
""")

st.success("Project Initialized Successfully")