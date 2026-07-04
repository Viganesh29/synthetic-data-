import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

TARGET_COLUMNS = [
    "Credit_Score",
    "Default_Risk",
    "Loan_Approved"
]

BINARY_TARGET_COLUMNS = [
    "Loan_Approved"
]

ORDINAL_TARGET_COLUMNS = {
    "Default_Risk": (0, 2)
}

def preprocess_data(df):

    df = df.copy()

    categorical_columns = list(
        df.select_dtypes(
            include=["object"]
        ).columns
    )

    encoders = {}

    for col in categorical_columns:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )

        encoders[col] = encoder

    feature_columns = [

        col

        for col in df.columns

        if col not in TARGET_COLUMNS

    ]

    scaler = MinMaxScaler()

    df[feature_columns] = scaler.fit_transform(
        df[feature_columns]
    )

    return df, encoders, scaler


def clean_synthetic_targets(df):

    df = df.copy()

    for col in BINARY_TARGET_COLUMNS:

        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):

            df[col] = (
                df[col]
                .clip(0, 1)
                .round()
                .astype(int)
            )

    for col, (min_value, max_value) in ORDINAL_TARGET_COLUMNS.items():

        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):

            df[col] = (
                df[col]
                .clip(min_value, max_value)
                .round()
                .astype(int)
            )

    return df


def align_synthetic_targets_to_reference(
        synthetic_df,
        reference_df
):

    synthetic_df = clean_synthetic_targets(
        synthetic_df
    )

    reference_df, _, _ = preprocess_data(
        reference_df
    )

    for col in [
        "Default_Risk",
        "Loan_Approved"
    ]:

        if col not in synthetic_df.columns or col not in reference_df.columns:

            continue

        if not pd.api.types.is_numeric_dtype(
            synthetic_df[col]
        ):

            continue

        synthetic_unique = synthetic_df[col].nunique(
            dropna=True
        )

        reference_counts = (
            reference_df[col]
            .value_counts(normalize=True)
            .sort_index()
        )

        if synthetic_unique >= min(
            2,
            len(reference_counts)
        ):

            continue

        synthetic_df[col] = _assign_classes_by_distribution(
            synthetic_df[col],
            reference_counts
        )

    return synthetic_df


def _assign_classes_by_distribution(
        scores,
        class_proportions
):

    scores = scores.fillna(
        scores.median()
    )

    n_rows = len(scores)

    raw_counts = class_proportions * n_rows

    class_counts = np.floor(
        raw_counts
    ).astype(int)

    remainder = n_rows - class_counts.sum()

    if remainder > 0:

        fractional_parts = (
            raw_counts - class_counts
        ).sort_values(
            ascending=False
        )

        for class_value in fractional_parts.index[:remainder]:

            class_counts.loc[class_value] += 1

    sorted_index = scores.sort_values().index

    assigned = pd.Series(
        index=scores.index,
        dtype=int
    )

    start = 0

    for class_value, count in class_counts.items():

        end = start + count

        assigned.loc[
            sorted_index[start:end]
        ] = int(class_value)

        start = end

    return assigned.astype(int)


def get_categorical_columns(df):

    return list(
        df.select_dtypes(
            include=["object"]
        ).columns
    )


def get_numerical_columns(df):

    return list(
        df.select_dtypes(
            exclude=["object"]
        ).columns
    )
