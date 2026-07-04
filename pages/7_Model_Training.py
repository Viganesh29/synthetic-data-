import streamlit as st
import pandas as pd
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

import matplotlib.pyplot as plt

from utils.data_loader import load_data
from utils.preprocessing import (
    align_synthetic_targets_to_reference,
    preprocess_data
)
from models.classifier import CreditRiskClassifier

st.title("🤖 Credit Risk Model Training")

# =====================================================
# DATASET SELECTION
# =====================================================

dataset_choice = st.selectbox(
    "Select Training Dataset",
    [
        "Original Dataset",
        "VAE Synthetic Dataset",
        "GAN Synthetic Dataset",
        "CTGAN Synthetic Dataset"
    ]
)

try:

    if dataset_choice == "Original Dataset":

        df = load_data()

    elif dataset_choice == "VAE Synthetic Dataset":

        df = pd.read_csv(
            "outputs/generated_data/synthetic_vae.csv"
        )

        df = align_synthetic_targets_to_reference(
            df,
            load_data()
        )

    elif dataset_choice == "GAN Synthetic Dataset":

        df = pd.read_csv(
            "outputs/generated_data/synthetic_gan.csv"
        )

        df = align_synthetic_targets_to_reference(
            df,
            load_data()
        )

    elif dataset_choice == "CTGAN Synthetic Dataset":

        df = pd.read_csv(
            "outputs/generated_data/synthetic_ctgan.csv"
        )

        df = align_synthetic_targets_to_reference(
            df,
            load_data()
        )

except Exception as e:

    st.error(
        f"Dataset loading error: {e}"
    )

    st.stop()

# =====================================================
# DATA PREVIEW
# =====================================================

st.subheader("Dataset Preview")

st.dataframe(
    df.head()
)

# =====================================================
# TARGET SELECTION
# =====================================================

available_targets = []

for col in [
    "Default_Risk",
    "Loan_Approved"
]:

    if col in df.columns:

        available_targets.append(col)

if len(available_targets) == 0:

    st.error(
        "No valid target columns found."
    )

    st.stop()

target_column = st.selectbox(
    "Select Target Column",
    available_targets
)

# =====================================================
# MODEL LABEL
# =====================================================

model_name = st.selectbox(
    "Save Metrics As",
    [
        "VAE",
        "GAN",
        "CTGAN"
    ]
)

# =====================================================
# TRAIN BUTTON
# =====================================================

if st.button("🚀 Train Model"):

    try:

        processed_df, encoders, scaler = preprocess_data(
            df
        )

        drop_columns = [
            "Application_ID",
            "Risk_Score",
            "Default_Risk",
            "Loan_Approved",
            "Credit_Score"
        ]

        if target_column in drop_columns:

            drop_columns.remove(
                target_column
            )

        X = processed_df.drop(
            columns=drop_columns,
            errors="ignore"
        )

        if target_column in X.columns:

            X = X.drop(
                columns=[target_column]
            )

        y = df[target_column]

        label_encoder = LabelEncoder()

        y = label_encoder.fit_transform(
            y.astype(str)
        )

        if len(label_encoder.classes_) < 2:

            st.error(
                "The selected target has only one class after cleanup. "
                "Regenerate the synthetic dataset or choose another target."
            )

            st.stop()

        st.subheader(
            "Target Distribution"
        )

        st.write(
            pd.Series(y).value_counts()
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = CreditRiskClassifier()

        model.train(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        results = model.evaluate(
            y_test,
            predictions
        )

        try:

            probabilities = model.model.predict_proba(
                X_test
            )

            if len(label_encoder.classes_) == 2:

                auc = roc_auc_score(
                    y_test,
                    probabilities[:, 1]
                )

            else:

                auc = roc_auc_score(
                    y_test,
                    probabilities,
                    multi_class="ovr"
                )

            results["AUROC"] = round(
                auc,
                4
            )

        except:

            results["AUROC"] = None

        st.success(
            "Model Training Completed"
        )

        st.subheader(
            "Performance Metrics"
        )

        st.json(
            results
        )

        # ====================================
        # SAVE METRICS
        # ====================================

        os.makedirs(
            "outputs/metrics",
            exist_ok=True
        )

        metrics_path = (
            f"outputs/metrics/"
            f"{model_name.lower()}_metrics.json"
        )

        with open(
            metrics_path,
            "w"
        ) as f:

            json.dump(
                results,
                f,
                indent=4
            )

        st.success(
            f"Metrics saved to {metrics_path}"
        )

        # ====================================
        # CONFUSION MATRIX
        # ====================================

        st.subheader(
            "Confusion Matrix"
        )

        cm = confusion_matrix(
            y_test,
            predictions
        )

        fig, ax = plt.subplots(
            figsize=(6, 6)
        )

        ConfusionMatrixDisplay(
            confusion_matrix=cm
        ).plot(ax=ax)

        st.pyplot(fig)

    except Exception as e:

        st.error(
            f"Training Error: {e}"
        )
