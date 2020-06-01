from sklearn.base import RegressorMixin

from autoflow.core.base import AutoFlowEstimator

__all__ = ["AutoFlowRegressor"]


class AutoFlowRegressor(AutoFlowEstimator, RegressorMixin):
    checked_mainTask = "regression"

    def predict(
            self,
            X_test
    ):
        self._predict(X_test)
        return self.estimator.predict(self.data_manager.X_test)
