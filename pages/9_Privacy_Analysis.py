import streamlit as st
import pandas as pd

from utils.data_loader import load_data

from utils.preprocessing import (
    preprocess_data
)

from utils.privacy import (
    calculate_dcr,
    privacy_risk_level,
    membership_inference_score
)

st.title(
    "🔐 Privacy Evaluation"
)

try:

    real_df = load_data()

    synthetic_df = pd.read_csv(
        "outputs/generated_data/synthetic_ctgan.csv"
    )

except:

    st.error(
        "Generate CTGAN data first."
    )

    st.stop()

real_df, _, _ = preprocess_data(
    real_df
)

synthetic_df = synthetic_df[
    real_df.columns
]

results = calculate_dcr(
    real_df,
    synthetic_df
)

risk_level = privacy_risk_level(
    results["DCR"]
)

membership_score = (
    membership_inference_score(
        real_df,
        synthetic_df
    )
)

st.header(
    "Distance To Closest Record"
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "DCR",
        round(
            results["DCR"],
            4
        )
    )

with col2:

    st.metric(
        "Average DCR",
        round(
            results[
                "Average_DCR"
            ],
            4
        )
    )

st.header(
    "Privacy Status"
)

if risk_level == "SAFE":

    st.success(
        "SAFE : DCR > 0.5"
    )

elif risk_level == "MODERATE":

    st.warning(
        "MODERATE : DCR between 0.3 and 0.5"
    )

else:

    st.error(
        "HIGH RISK : DCR < 0.3"
    )

st.header(
    "Membership Inference Risk"
)

st.metric(
    "Risk %",
    membership_score
)

st.info(
    """
    Lower is Better

    Less than 10%
    indicates strong privacy.
    """
)