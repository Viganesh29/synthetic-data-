import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data

from utils.preprocessing import (
    preprocess_data
)

from utils.evaluation import (
    ks_test_analysis,
    wasserstein_analysis,
    correlation_preservation
)

st.title(
    "📊 Statistical Fidelity Analysis"
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

numeric_cols = real_df.columns

synthetic_df = synthetic_df[
    numeric_cols
]

st.header(
    "1️⃣ KS Test"
)

ks_results = ks_test_analysis(
    real_df,
    synthetic_df
)

st.dataframe(
    ks_results
)

st.header(
    "2️⃣ Wasserstein Distance"
)

wasserstein_results = wasserstein_analysis(
    real_df,
    synthetic_df
)

st.dataframe(
    wasserstein_results
)

st.header(
    "3️⃣ Correlation Preservation"
)

corr_results = correlation_preservation(
    real_df,
    synthetic_df
)

st.metric(
    "Average Correlation Difference",
    round(
        corr_results[
            "Average Difference"
        ],
        4
    )
)

st.subheader(
    "Real Data Correlation"
)

fig1 = px.imshow(
    corr_results[
        "Real Correlation"
    ],
    text_auto=True
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

st.subheader(
    "Synthetic Data Correlation"
)

fig2 = px.imshow(
    corr_results[
        "Synthetic Correlation"
    ],
    text_auto=True
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

st.header(
    "4️⃣ Distribution Comparison"
)

selected_feature = st.selectbox(
    "Choose Feature",
    real_df.columns
)

comparison_df = pd.DataFrame({

    "Real":
    real_df[
        selected_feature
    ],

    "Synthetic":
    synthetic_df[
        selected_feature
    ]

})

fig = px.histogram(

    comparison_df,

    barmode="overlay",

    opacity=0.6

)

st.plotly_chart(
    fig,
    use_container_width=True
)