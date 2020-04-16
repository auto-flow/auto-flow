from sklearn.multiclass import OneVsRestClassifier

from autoflow.pipeline.components.base import AutoFlowComponent
from autoflow.utils.data import softmax, densify
from autoflow.utils.ml_task import get_ml_task_from_y


class AutoFlowClassificationAlgorithm(AutoFlowComponent):
    """Provide an abstract interface for classification algorithms in
    auto-sklearn.

    See :ref:`extending` for more information."""

    OVR__: bool = False

    def isOVR(self):
        return self.OVR__

    def after_process_estimator(self, estimator, X_train, y_train=None, X_valid=None, y_valid=None, X_test=None,
                                y_test=None):
        # def after_process_estimator(self, estimator, X, y):
        if self.isOVR() and get_ml_task_from_y(y_train).subTask != "binary":
            estimator = OneVsRestClassifier(estimator, n_jobs=1)
        return estimator

    # def _pred_or_trans(self, X_train, X_valid=None, X_test=None):
    def _pred_or_trans(self, X_train_, X_valid_=None, X_test_=None, X_train=None, X_valid=None, X_test=None,
                       y_train=None):
        return self.estimator.predict(self.before_pred_X(X_train_))

    def predict(self, X):
        return self.pred_or_trans(X)

    def predict_proba(self, X):
        X = self.preprocess_data(X)
        if not self.estimator:
            raise NotImplementedError()
        if not hasattr(self, "predict_proba"):
            if hasattr(self, "decision_function"):
                df = self.estimator.decision_function(self.before_pred_X(X))
                return softmax(df)
            else:
                raise NotImplementedError()
        return self.estimator.predict_proba(self.before_pred_X(X))

    def score(self, X, y):
        X = densify(X)
        return self.estimator.score(X, y)
