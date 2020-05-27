from copy import deepcopy
from typing import Dict

import numpy as np

from autoflow.manager.data_container.dataframe import DataFrameContainer
from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm
from autoflow.workflow.components.utils import get_categorical_features_indices

__all__ = ["CatBoostRegressor"]


class CatBoostRegressor(AutoFlowClassificationAlgorithm):
    class__ = "CatBoostRegressor"
    module__ = "catboost"

    boost_model = True
    tree_model = True

    def core_fit(self, estimator, X, y=None, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None):
        categorical_features_indices = None #get_categorical_features_indices(X)
        if (X_valid is not None) and (y_valid is not None):
            eval_set = (X_valid, y_valid)
        else:
            eval_set = None
        return self.component.fit(
            X, y, cat_features=categorical_features_indices,
            eval_set=eval_set, silent=True
        )

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams = deepcopy(hyperparams)
        if "n_jobs" in hyperparams:
            hyperparams["thread_count"] = hyperparams.pop("n_jobs")
        return hyperparams

    def before_fit_X(self, X: DataFrameContainer):
        X = super(CatBoostRegressor, self).before_fit_X(X)
        if X is None:
            return None
        return np.array(X)
