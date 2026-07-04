import pandas as pd
import numpy as np


def demographic_parity_difference(
        df,
        group_column,
        target_column
):

    grouped = df.groupby(
        group_column
    )[target_column].mean()

    max_rate = grouped.max()

    min_rate = grouped.min()

    dpd = abs(
        max_rate - min_rate
    )

    return {

        "DPD": float(dpd),

        "Group Rates":
        grouped.to_dict()
    }


def fairness_score(dpd):

    return max(
        0,
        (1 - dpd) * 100
    )


def bias_level(dpd):

    if dpd <= 0.10:

        return "LOW"

    elif dpd <= 0.25:

        return "MODERATE"

    else:

        return "HIGH"