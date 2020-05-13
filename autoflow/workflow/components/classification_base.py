from sklearn.base import ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier

from autoflow.manager.data_container.dataframe import DataFrameContainer
from autoflow.workflow.components.base import AutoFlowComponent
from autoflow.utils.data import softmax
from autoflow.utils.ml_task import get_ml_task_from_y


class AutoFlowClassificationAlgorithm(AutoFlowComponent, ClassifierMixin):
    """Provide an abstract interface for classification algorithms in
    auto-sklearn.

    See :ref:`extending` for more information."""

    OVR__: bool = False

    def isOVR(self):
        return self.OVR__

    def after_process_estimator(self, estimator, X_train, y_train=None, X_valid=None, y_valid=None, X_test=None,
                                y_test=None):
        if self.isOVR() and get_ml_task_from_y(y_train).subTask != "binary":
            estimator = OneVsRestClassifier(estimator, n_jobs=1)
        return estimator

    def predict(self, X):
        return self.component.predict(self.before_pred_X(X))

    def predict_proba(self, X: DataFrameContainer):
        X = self.filter_feature_groups(X)
        if not self.component:
            raise NotImplementedError()
        if not hasattr(self, "predict_proba"):
            if hasattr(self, "decision_function"):
                df = self.component.decision_function(self.before_pred_X(X))
                return softmax(df)
            else:
                raise NotImplementedError()
        return self.component.predict_proba(self.before_pred_X(X))
