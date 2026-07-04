import streamlit as st

from utils.data_loader import load_data
from utils.preprocessing import (
preprocess_data,
get_categorical_columns,
get_numerical_columns
)

st.title("⚙️ Data Preprocessing")

df = load_data()

st.subheader("Original Dataset")

st.write(df.head())

st.write(df.shape)

categorical_cols = get_categorical_columns(df)
numerical_cols = get_numerical_columns(df)

st.subheader("Categorical Columns")

st.write(categorical_cols)

st.subheader("Numerical Columns")

st.write(numerical_cols)

processed_df, encoders, scaler = preprocess_data(df)

st.subheader("Processed Dataset")

st.write(processed_df.head())

st.write(processed_df.shape)

st.success("Preprocessing Completed Successfully")
