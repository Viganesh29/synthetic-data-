import streamlit as st

from utils.data_loader import (
    load_data,
    get_dataset_info
)

from utils.visualizations import (
    class_distribution_chart,
    correlation_heatmap
)

st.title("📊 Dataset Analysis")

df = load_data()

st.dataframe(df.head())

info = get_dataset_info(df)

col1, col2, col3 = st.columns(3)

col1.metric("Rows", info["Rows"])
col2.metric("Columns", info["Columns"])
col3.metric("Missing Values", info["Missing Values"])

if "Default_Risk" in df.columns:

    st.plotly_chart(
        class_distribution_chart(
            df,
            "Default_Risk"
        ),
        use_container_width=True
    )

st.plotly_chart(
    correlation_heatmap(df),
    use_container_width=True
)
