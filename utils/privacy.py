import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances


def calculate_dcr(
        real_df,
        synthetic_df
):

    distances = euclidean_distances(
        synthetic_df,
        real_df
    )

    min_distances = np.min(
        distances,
        axis=1
    )

    dcr = np.min(
        min_distances
    )

    avg_dcr = np.mean(
        min_distances
    )

    return {

        "DCR": float(dcr),

        "Average_DCR":
        float(avg_dcr)

    }


def privacy_risk_level(dcr):

    if dcr > 0.5:

        return "SAFE"

    elif dcr > 0.3:

        return "MODERATE"

    else:

        return "HIGH RISK"


def membership_inference_score(
        real_df,
        synthetic_df
):

    distances = euclidean_distances(
        synthetic_df,
        real_df
    )

    min_distances = np.min(
        distances,
        axis=1
    )

    threshold = np.percentile(
        min_distances,
        10
    )

    risky_records = np.sum(
        min_distances < threshold
    )

    risk_percentage = (
        risky_records
        /
        len(min_distances)
    ) * 100

    return round(
        risk_percentage,
        2
    )