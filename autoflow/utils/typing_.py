from typing_extensions import Protocol


class GenericEstimator(Protocol):
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass
