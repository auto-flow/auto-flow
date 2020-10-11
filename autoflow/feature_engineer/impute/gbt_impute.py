"""MissForest Imputer for Missing Data"""
# Author: Ashim Bhattarai
# License: GNU General Public License v3 (GPLv3)

from autoflow.estimator.wrap_lightgbm import LGBMClassifier, LGBMRegressor

from .predictive import PredictiveImputer

__all__ = [
    'GBTImputer',
]


class GBTImputer(PredictiveImputer):
    reg_cls = LGBMRegressor
    clf_cls = LGBMClassifier

    def __init__(
            self,
            categorical_feature=None,
            numerical_feature=None,
            copy=False,
            missing_rate=0.4,
            max_iter=10,
            decreasing=False,
            budget=10,
            verbose=0,
            n_estimators=100,
            learning_rate=0.01,
            random_state=42,
            n_jobs=-1
    ):
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.budget = budget
        self.decreasing = decreasing
        self.max_iter = max_iter
        self.copy = copy
        self.numerical_feature = numerical_feature
        self.categorical_feature = categorical_feature
        super(GBTImputer, self).__init__(
            categorical_feature=categorical_feature,
            numerical_feature=numerical_feature,
            copy=copy,
            missing_rate=missing_rate,
            max_iter=max_iter,
            decreasing=decreasing,
            budget=budget,
            verbose=verbose,
            params=dict(
                random_state=random_state,
                n_jobs=n_jobs,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
            )
        )

    def update_params(self, params, problem_type):
        return params
