from functools import partial
from typing import Dict

from sklearn.ensemble import IsolationForest as SklearnIsolationForest

from autoflow.pipeline.components.data_process_base import AutoFlowDataProcessAlgorithm

__all__ = ["IsolationForest"]


def outlier_rejection(X, y, hyperparams):
    model = SklearnIsolationForest(
        max_samples=hyperparams.get("max_samples", 100),
        contamination=hyperparams.get("contamination", 0.4),
        random_state=hyperparams.get("random_state", 42)
    )
    model.fit(X)
    y_pred = model.predict(X)
    return X[y_pred == 1], y[y_pred == 1]


class IsolationForest(AutoFlowDataProcessAlgorithm):
    class__ = "FunctionSampler"
    module__ = "imblearn"

    def after_process_hyperparams(self, hyperparams: dict) -> Dict:
        hyperparams = super(IsolationForest, self).after_process_hyperparams(hyperparams)

        hyperparams["func"] = partial(outlier_rejection, hyperparams=hyperparams)
        return hyperparams
