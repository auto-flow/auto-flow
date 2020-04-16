from copy import deepcopy
from typing import Optional

from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm
from autoflow.pipeline.dataframe import GenericDataFrame

__all__ = ["Merge"]


class Merge(AutoFlowFeatureEngineerAlgorithm):

    def fit(self, X_train: GenericDataFrame, y_train=None,
            X_valid=None, y_valid=None,
            X_test=None, y_test=None):
        return self

    def process(self, X_origin: Optional[GenericDataFrame]) -> Optional[GenericDataFrame]:
        X = deepcopy(X_origin)
        if X is None:
            return None
        if isinstance(self.out_feature_groups, (list, tuple)):
            assert len(self.out_feature_groups) == 1
            self.out_feature_groups = self.out_feature_groups[0]
        # fixme: deepcopy
        X.set_feature_groups(X.feature_groups.replace(self.in_feature_groups, self.out_feature_groups))
        return X

    def transform(self, X_train=None, X_valid=None, X_test=None, y_train=None):
        return {
            "X_train": self.process(X_train),
            "X_valid": self.process(X_valid),
            "X_test": self.process(X_test),
            "y_train": y_train
        }
