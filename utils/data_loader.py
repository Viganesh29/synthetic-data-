import pandas as pd


DATA_PATH = "data/loan_approval_dataset.csv"


def load_data():

    df = pd.read_csv(DATA_PATH)

    return df


def get_dataset_info(df):

    info = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": df.isnull().sum().sum()
    }

    return info