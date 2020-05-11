from typing import Optional

from autoflow.manager.data_container.dataframe import DataFrameContainer
from autoflow.manager.data_container.base import copy_data_container_structure
from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["Merge"]


class Merge(AutoFlowFeatureEngineerAlgorithm):

    def fit(self, X_train: DataFrameContainer, y_train=None,
            X_valid=None, y_valid=None,
            X_test=None, y_test=None):
        return self

    def process(self, X: Optional[DataFrameContainer]) -> Optional[DataFrameContainer]:
        if X is None:
            return None
        if isinstance(self.out_feature_groups, (list, tuple)):
            assert len(self.out_feature_groups) == 1
            self.out_feature_groups = self.out_feature_groups[0]
        X_ = copy_data_container_structure(X)
        X_.data = X.data
        X_.set_feature_groups(X_.feature_groups.replace(self.in_feature_groups, self.out_feature_groups))
        return X_

    def transform(self, X_train=None, X_valid=None, X_test=None, y_train=None):
        return {
            "X_train": self.process(X_train),
            "X_valid": self.process(X_valid),
            "X_test": self.process(X_test),
            "y_train": y_train
        }
