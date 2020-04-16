from copy import deepcopy
from typing import Dict

from autoflow.pipeline.components.classification_base import AutoFlowClassificationAlgorithm
from autoflow.pipeline.components.utils import get_categorical_features_indices

__all__ = ["CatBoostRegressor"]


class CatBoostRegressor(AutoFlowClassificationAlgorithm):
    class__ = "CatBoostRegressor"
    module__ = "catboost"

    boost_model = True
    tree_model = True

    def core_fit(self, estimator, X, y, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None, columns_metadata=None):
        categorical_features_indices = get_categorical_features_indices(X, columns_metadata)
        if (X_valid is not None) and (y_valid is not None):
            eval_set = (X_valid, y_valid)
        else:
            eval_set = None
        return self.estimator.fit(
            X, y, cat_features=categorical_features_indices,
            eval_set=eval_set, silent=True
        )

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams = deepcopy(hyperparams)
        if "n_jobs" in hyperparams:
            hyperparams["thread_count"] = hyperparams.pop("n_jobs")
        return hyperparams
