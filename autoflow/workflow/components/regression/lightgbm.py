import numpy as np

from autoflow.workflow.components.iter_algo import LgbmIterativeMixIn
from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm

__all__ = ["LGBMRegressor"]


class LGBMRegressor(AutoFlowRegressionAlgorithm, LgbmIterativeMixIn):
    class__ = "LGBMRegressor"
    module__ = "autoflow.estimator.wrap_lightgbm"

    boost_model = True
    tree_model = True
    support_early_stopping = True

    def core_fit(self, estimator, X, y=None, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None, **kwargs):
        use_categorical_feature = self.hyperparams.get("use_categorical_feature", False)
        categorical_features_indices = np.arange(len(feature_groups))[feature_groups == "ordinal"].tolist()
        if not use_categorical_feature:
            categorical_features_indices = "auto"
        component = self.component.fit(
            X, y, X_valid, y_valid, categorical_feature=categorical_features_indices,
            sample_weight=kwargs.get("sample_weight")
        )
        self.best_iteration_ = component.model.best_iteration
        return component
