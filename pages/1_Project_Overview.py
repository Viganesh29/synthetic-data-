import streamlit as st

st.title("🏦 Synthetic Financial Data Generation")

st.markdown("""

## Research Overview

This project develops a Generative AI framework for
synthetic financial data generation to support:

* Credit Risk Assessment
* Privacy Preservation
* Fairness Analysis
* Regulatory Testing

---

### Models Used

1. Variational Autoencoder (VAE)

2. Generative Adversarial Network (GAN)

3. Conditional Tabular GAN (CTGAN)

---

### Evaluation Metrics

#### Performance

* Accuracy
* Precision
* Recall
* F1 Score
* AUROC

#### Statistical Fidelity

* KS Test
* Wasserstein Distance
* Correlation Preservation

#### Privacy

* DCR
* Membership Inference Risk

#### Fairness

* Demographic Parity Difference

---

### Workflow

Dataset
↓
Preprocessing
↓
VAE / GAN / CTGAN
↓
Synthetic Data
↓
Credit Risk Model
↓
Evaluation
""")
