from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

class CreditRiskClassifier:

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_true, y_pred):

        return {
            "Accuracy": round(
                accuracy_score(y_true, y_pred),
                4
            ),

            "Precision": round(
                precision_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                ),
                4
            ),

            "Recall": round(
                recall_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                ),
                4
            ),

            "F1": round(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                ),
                4
            )
        }