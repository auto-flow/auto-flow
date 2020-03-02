from sklearn.multiclass import OneVsRestClassifier

from autopipeline.pipeline.components.base import AutoPLComponent
from autopipeline.utils.data import get_task_from_y, softmax, densify


class AutoPLClassificationAlgorithm(AutoPLComponent):
    """Provide an abstract interface for classification algorithms in
    auto-sklearn.

    See :ref:`extending` for more information."""

    OVR__: bool = False

    def isOVR(self):
        return self.OVR__

    def after_process_estimator(self, estimator, X, y):
        if self.isOVR() and get_task_from_y(y).subTask != "binary":
            estimator = OneVsRestClassifier(estimator, n_jobs=1)
        return estimator

    def after_process_pred_y(self, y):
        return y

    def predict(self, X):
        X = densify(X)
        if not self.estimator:
            raise NotImplementedError()
        pred_y = self.estimator.predict(X)
        return self.after_process_pred_y(pred_y)

    def predict_proba(self, X):
        X = densify(X)
        if not self.estimator:
            raise NotImplementedError()
        if not hasattr(self, "predict_proba"):
            if hasattr(self, "decision_function"):
                df = self.estimator.decision_function(X)
                return softmax(df)
            else:
                raise NotImplementedError()
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        X = densify(X)
        return self.estimator.score(X, y)
