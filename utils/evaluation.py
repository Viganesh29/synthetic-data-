import pandas as pd
import numpy as np

from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance


def ks_test_analysis(real_df, synthetic_df):

    results = []

    common_columns = [
        col for col in real_df.columns
        if col in synthetic_df.columns
    ]

    for col in common_columns:

        try:

            stat, p_value = ks_2samp(
                real_df[col],
                synthetic_df[col]
            )

            results.append({

                "Feature": col,
                "KS Statistic": stat,
                "P Value": p_value

            })

        except:

            pass

    return pd.DataFrame(results)


def wasserstein_analysis(
        real_df,
        synthetic_df
):

    results = []

    common_columns = [
        col for col in real_df.columns
        if col in synthetic_df.columns
    ]

    for col in common_columns:

        try:

            wd = wasserstein_distance(
                real_df[col],
                synthetic_df[col]
            )

            results.append({

                "Feature": col,
                "Wasserstein Distance": wd

            })

        except:

            pass

    return pd.DataFrame(results)


def correlation_preservation(
        real_df,
        synthetic_df
):

    real_corr = real_df.corr()

    synthetic_corr = synthetic_df.corr()

    diff = np.abs(
        real_corr -
        synthetic_corr
    )

    score = diff.mean().mean()

    return {

        "Real Correlation": real_corr,
        "Synthetic Correlation": synthetic_corr,
        "Average Difference": score
    }