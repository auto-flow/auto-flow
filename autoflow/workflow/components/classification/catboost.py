from copy import deepcopy
from typing import Dict

from autoflow.data_container import DataFrameContainer
from autoflow.workflow.components.base import BoostingModelMixin
from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm
import numpy as np

__all__ = ["CatBoostClassifier"]


class CatBoostClassifier(AutoFlowClassificationAlgorithm, BoostingModelMixin):
    class__ = "CatBoostClassifier"
    module__ = "catboost"

    boost_model = True
    tree_model = True
    support_early_stopping = True

    def core_fit(self, estimator, X, y=None, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None,**kwargs):
        categorical_features_indices = None # get_categorical_features_indices(X)
        if (X_valid is not None) and (y_valid is not None):
            eval_set = (X_valid, y_valid)
        else:
            eval_set = None
        return self.component.fit(
            X, y, cat_features=categorical_features_indices,
            eval_set=eval_set, silent=True, **kwargs
        )

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams = deepcopy(hyperparams)
        if "n_jobs" in hyperparams:
            hyperparams["thread_count"] = hyperparams.pop("n_jobs")
        return hyperparams

    def before_fit_X(self, X: DataFrameContainer):
        X = super(CatBoostClassifier, self).before_fit_X(X)
        if X is None:
            return None
        return np.array(X)
