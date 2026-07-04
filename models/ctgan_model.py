from ctgan import CTGAN
import joblib


class CTGANGenerator:

    def __init__(self):

        self.model = CTGAN(
            epochs=10
        )

    def train(
        self,
        df,
        categorical_columns
    ):

        self.model.fit(
            df,
            categorical_columns
        )

    def generate(
        self,
        n_samples
    ):

        return self.model.sample(
            n_samples
        )

    def save(
        self,
        path
    ):

        joblib.dump(
            self.model,
            path
        )

    def load(
        self,
        path
    ):

        self.model = joblib.load(path)