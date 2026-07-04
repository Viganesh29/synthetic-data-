import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.preprocessing import preprocess_data

from utils.fairness import (
    demographic_parity_difference,
    fairness_score,
    bias_level
)

st.title(
    "⚖️ Fairness Analysis"
)

df = load_data()

st.write(
    "Dataset Shape:",
    df.shape
)

available_columns = df.columns.tolist()

group_column = st.selectbox(
    "Select Group Column",
    available_columns
)

target_column = st.selectbox(
    "Select Target Column",
    available_columns
)

if st.button(
    "Run Fairness Analysis"
):

    processed_df, _, _ = preprocess_data(
        df
    )

    results = demographic_parity_difference(
        processed_df,
        group_column,
        target_column
    )

    score = fairness_score(
        results["DPD"]
    )

    bias = bias_level(
        results["DPD"]
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "DPD",
            round(
                results["DPD"],
                4
            )
        )

    with col2:

        st.metric(
            "Fairness Score",
            round(
                score,
                2
            )
        )

    with col3:

        st.metric(
            "Bias Level",
            bias
        )

    group_df = pd.DataFrame({

        "Group":
        list(
            results[
                "Group Rates"
            ].keys()
        ),

        "Rate":
        list(
            results[
                "Group Rates"
            ].values()
        )
    })

    fig = px.bar(

        group_df,

        x="Group",

        y="Rate",

        title="Group Outcome Rates"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )