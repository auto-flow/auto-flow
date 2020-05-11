from copy import deepcopy
from typing import Optional

import pandas as pd

from autoflow.manager.data_container.dataframe import DataFrameContainer
from autoflow.workflow.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["Split"]


class Split(AutoFlowFeatureEngineerAlgorithm):

    def fit(self, X_train: DataFrameContainer, y_train=None,
            X_valid=None, y_valid=None,
            X_test=None, y_test=None):
        self.column2fg = self.hyperparams["column2fg"]
        return self

    def process(self, X_origin: Optional[DataFrameContainer]) -> Optional[DataFrameContainer]:
        X = deepcopy(X_origin)
        if X is None:
            return None
        if isinstance(self.out_feature_groups, (list, tuple)):
            assert len(self.out_feature_groups) == 1
            self.out_feature_groups = self.out_feature_groups[0]
        # fixme: deepcopy
        columns = X.columns
        feature_groups = X.feature_groups
        result = []
        for column, feature_group in zip(columns, feature_groups):
            if column in self.column2fg:
                result.append(self.column2fg[column])
            else:
                result.append(feature_group)

        X.set_feature_groups(pd.Series(result))
        return X

    def transform(self, X_train=None, X_valid=None, X_test=None, y_train=None):
        return {
            "X_train": self.process(X_train),
            "X_valid": self.process(X_valid),
            "X_test": self.process(X_test),
            "y_train": y_train
        }
