from typing import Optional, Dict

from sklearn.base import RegressorMixin

from autoflow.estimator.base import AutoFlowEstimator


class AutoFlowRegressor(AutoFlowEstimator, RegressorMixin):
    checked_mainTask = "regression"

    def predict(
            self,
            X_test,
            task_id=None,
            trial_id=None,
            experiment_id=None,
            column_descriptions: Optional[Dict] = None,
            highR_nan_threshold=0.5
    ):
        self._predict(X_test, task_id, trial_id, experiment_id, column_descriptions, highR_nan_threshold)
        return self.estimator.predict(self.data_manager.X_test)
