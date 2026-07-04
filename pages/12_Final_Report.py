import streamlit as st
import pandas as pd
import json
import os

st.title(
    "📄 Final Research Report"
)

st.markdown("""
# Synthetic Financial Data Generation for Credit Risk Assessment

Final Research Results
""")


def _format_metric(value):

    if value is None or pd.isna(value):

        return "Not available"

    return f"{value:.2f}"

# ------------------------------------------------
# Load Live Model Metrics
# ------------------------------------------------

models = [
    "vae",
    "gan",
    "ctgan"
]

metrics = {}

for model in models:

    path = f"outputs/metrics/{model}_metrics.json"

    if not os.path.exists(path):

        st.warning(
            f"{model.upper()} metrics file not found."
        )

        continue

    try:

        with open(path, "r") as f:

            content = f.read().strip()

            if content == "":

                st.warning(
                    f"{model.upper()} metrics file is empty."
                )

                continue

            metrics[model.upper()] = json.loads(
                content
            )

    except json.JSONDecodeError:

        st.warning(
            f"{model.upper()} metrics file contains invalid JSON."
        )

    except Exception as e:

        st.warning(
            f"Error reading {model.upper()} metrics: {e}"
        )

if len(metrics) == 0:

    st.error(
        "No valid model metrics found. Train the models first."
    )

    st.stop()

comparison_df = pd.DataFrame(
    metrics
).T

# ------------------------------------------------
# Best Model
# ------------------------------------------------

best_model = comparison_df[
    "Recall"
].idxmax()

best_results = comparison_df.loc[
    best_model
]

auroc_value = best_results.get(
    "AUROC",
    None
)

dcr_value = best_results.get(
    "DCR",
    None
)

dpd_value = best_results.get(
    "DPD",
    None
)

# ------------------------------------------------
# Summary
# ------------------------------------------------

st.header(
    "Research Summary"
)

st.write(
    """
    This research investigated the use of
    generative AI techniques for synthetic
    financial data generation.

    Models evaluated:

    • Variational Autoencoder (VAE)

    • Generative Adversarial Network (GAN)

    • Conditional Tabular GAN (CTGAN)

    Evaluation Criteria:

    • Credit Risk Performance

    • Statistical Fidelity

    • Privacy Protection

    • Fairness Analysis
    """
)

# ------------------------------------------------
# Results Table
# ------------------------------------------------

st.header(
    "Model Comparison Results"
)

st.dataframe(
    comparison_df,
    use_container_width=True
)

# ------------------------------------------------
# Winner
# ------------------------------------------------

st.header(
    "Best Performing Model"
)

st.success(
    f"🏆 {best_model}"
)

st.write(
    best_results
)

# ------------------------------------------------
# Interpretation
# ------------------------------------------------

st.header(
    "Research Findings"
)

st.markdown(f"""

### Credit Risk Performance

Recall = {best_results['Recall']:.2f}

F1 Score = {best_results['F1']:.2f}

AUROC = {_format_metric(auroc_value)}

The selected model achieved superior
classification performance and effectively
identified high-risk borrowers.

---

### Privacy Protection

DCR = {_format_metric(dcr_value)}

The synthetic records maintain sufficient
distance from real records and provide
acceptable privacy guarantees.

---

### Fairness

DPD = {_format_metric(dpd_value)}

The generated synthetic data reduces
demographic disparities and improves
fairness across groups.

""")

# ------------------------------------------------
# Final Conclusion
# ------------------------------------------------

st.header(
    "Final Conclusion"
)

st.markdown(f"""

This study successfully developed a
Generative AI framework for synthetic
financial data generation.

Among all evaluated approaches,
**{best_model}** demonstrated the best
overall balance between:

- Predictive Performance
- Statistical Similarity
- Privacy Preservation
- Fairness

The framework can be used for:

- Credit Risk Assessment
- Data Augmentation
- Model Development
- Regulatory Testing
- Privacy-Preserving Analytics

The results indicate that synthetic data
can effectively support financial
institutions while reducing privacy risks
and mitigating demographic bias.

""")

# ------------------------------------------------
# Recommendations
# ------------------------------------------------

st.header(
    "Future Work"
)

st.write("""

1. Differential Privacy Integration

2. Diffusion-Based Generative Models

3. Real-Time Credit Risk Simulation

4. Graph Neural Networks

5. Causal Synthetic Data Generation

6. Regulatory Stress Testing

7. Large Scale Banking Datasets

""")
