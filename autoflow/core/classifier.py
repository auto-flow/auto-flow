from sklearn.base import ClassifierMixin

from autoflow.core.base import AutoFlowEstimator

__all__ = ["AutoFlowClassifier"]


class AutoFlowClassifier(AutoFlowEstimator, ClassifierMixin):
    checked_mainTask = "classification"

    def predict(
            self,
            X_test
    ):
        self._predict(X_test)
        return self.estimator.predict(self.data_manager.X_test)

    def predict_proba(
            self,
            X_test
    ):
        self._predict(X_test)
        return self.estimator.predict_proba(self.data_manager.X_test)

    def score(self, X, y, sample_weight=None):
        y=self.data_manager.encode_label(y)
        return super(AutoFlowClassifier, self).score(X, y)
