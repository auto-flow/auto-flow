from typing import Optional, Dict

from sklearn.base import RegressorMixin

from autoflow.estimator.base import AutoFlowEstimator


class AutoFlowRegressor(AutoFlowEstimator, RegressorMixin):
    checked_mainTask = "regression"

    def predict(
            self,
            X_test
    ):
        self._predict(X_test)
        return self.estimator.predict(self.data_manager.X_test)
