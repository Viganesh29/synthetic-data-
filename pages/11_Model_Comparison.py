import streamlit as st
import pandas as pd
import json
import os

st.title("🏆 Model Comparison Dashboard")

models = [
    "vae",
    "gan",
    "ctgan"
]

comparison = []

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

            metrics = json.loads(content)

        metrics["Model"] = model.upper()

        comparison.append(metrics)

    except json.JSONDecodeError:

        st.warning(
            f"{model.upper()} metrics file contains invalid JSON."
        )

        continue

    except Exception as e:

        st.warning(
            f"Error reading {model.upper()} metrics: {e}"
        )

        continue

if len(comparison) == 0:

    st.error(
        """
        No valid metrics found.

        Train a model first and save
        VAE / GAN / CTGAN metrics.
        """
    )

    st.stop()

comparison_df = pd.DataFrame(
    comparison
)

st.subheader(
    "Model Performance Table"
)

st.dataframe(
    comparison_df,
    use_container_width=True
)

# -----------------------------
# Best Model Selection
# -----------------------------

if "Recall" in comparison_df.columns:

    best_model = comparison_df.loc[
        comparison_df["Recall"].idxmax()
    ]

    st.success(
        f"🏆 Best Model: {best_model['Model']}"
    )

    st.subheader(
        "Best Model Metrics"
    )

    st.write(
        best_model
    )

# -----------------------------
# Charts
# -----------------------------

metric_columns = [

    col

    for col in comparison_df.columns

    if col != "Model"

]

for metric in metric_columns:

    st.subheader(
        f"{metric} Comparison"
    )

    chart_df = comparison_df[
        ["Model", metric]
    ]

    st.bar_chart(
        chart_df.set_index(
            "Model"
        )
    )

    st.write(comparison_df)