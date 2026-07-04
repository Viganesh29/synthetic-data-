import plotly.express as px
import pandas as pd


def class_distribution_chart(df, column):

    fig = px.histogram(
        df,
        x=column,
        title=f"{column} Distribution"
    )

    return fig


def correlation_heatmap(df):

    numeric_df = df.select_dtypes(
        exclude=["object"]
    )

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )

    return fig


def missing_values_chart(df):

    missing = df.isnull().sum()

    fig = px.bar(
        x=missing.index,
        y=missing.values,
        labels={
            "x": "Features",
            "y": "Missing Count"
        },
        title="Missing Values"
    )

    return fig