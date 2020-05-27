from typing import Optional, Dict

from sklearn.base import ClassifierMixin

from autoflow.estimator.base import AutoFlowEstimator


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
