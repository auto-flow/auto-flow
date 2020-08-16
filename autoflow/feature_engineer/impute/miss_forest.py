"""MissForest Imputer for Missing Data"""
# Author: Ashim Bhattarai
# License: GNU General Public License v3 (GPLv3)
from copy import deepcopy

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .predictive import PredictiveImputer

__all__ = [
    'MissForest',
]


class MissForest(PredictiveImputer):
    reg_cls = RandomForestRegressor
    clf_cls = RandomForestClassifier

    def __init__(
            self,
            categorical_feature=None,
            numerical_feature=None,
            copy=False,
            max_iter=10,
            decreasing=False,
            budget=10,
            verbose=0,
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features='auto',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            random_state=42,
    ):

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.oob_score = oob_score
        self.bootstrap = bootstrap
        self.min_impurity_decrease = min_impurity_decrease
        self.max_leaf_nodes = max_leaf_nodes
        self.max_features = max_features
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.budget = budget
        self.decreasing = decreasing
        self.max_iter = max_iter
        self.copy = copy
        self.numerical_feature = numerical_feature
        self.categorical_feature = categorical_feature
        super(MissForest, self).__init__(
            categorical_feature=categorical_feature,
            numerical_feature=numerical_feature,
            copy=copy,
            max_iter=max_iter,
            decreasing=decreasing,
            budget=budget,
            verbose=verbose,
            params=dict(
                random_state=random_state,
                n_jobs=n_jobs,
                oob_score=oob_score,
                bootstrap=bootstrap,
                min_impurity_decrease=min_impurity_decrease,
                max_leaf_nodes=max_leaf_nodes,
                max_features=max_features,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                max_depth=max_depth,
                n_estimators=n_estimators,
            )
        )

    def update_params(self, params, problem_type):
        params = deepcopy(params)
        if problem_type == "classification":
            criterion = "gini"
        elif problem_type == "regression":
            criterion = "mse"
        else:
            raise ValueError(f"Unknown problem-type {problem_type}")
        params["criterion"] = criterion
        return params
