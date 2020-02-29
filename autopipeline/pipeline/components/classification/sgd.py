import numpy as np

from autopipeline.pipeline.components.classification_base import AutoPLClassificationAlgorithm
from autopipeline.utils.data import softmax

__all__=["SGD"]


class SGD(
    AutoPLClassificationAlgorithm
):
    module__ = "sklearn.linear_model.stochastic_gradient"
    class__ = "SGDClassifier"

    def predict_proba(self, X):
        if self.hyperparams["loss"] in ["log", "modified_huber"]:
            return super(SGD, self).predict_proba(X)
        else:
            df = self.estimator.decision_function(X)
            return softmax(df)

