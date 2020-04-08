from typing import Optional

from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm
from hyperflow.pipeline.dataframe import GenericDataFrame

__all__ = ["Merge"]


class Merge(HyperFlowFeatureEngineerAlgorithm):

    def fit(self, X_train: GenericDataFrame, y_train=None,
            X_valid=None, y_valid=None,
            X_test=None, y_test=None):
        return self

    def process(self, X_origin: Optional[GenericDataFrame]) -> Optional[GenericDataFrame]:

        if X_origin is None:
            return None
        if isinstance(self.out_feature_groups, (list, tuple)):
            assert len(self.out_feature_groups) == 1
            self.out_feature_groups = self.out_feature_groups[0]
        # fixme: deepcopy
        X_origin.set_feature_groups(X_origin.feature_groups.replace(self.in_feature_groups, self.out_feature_groups))
        return X_origin

    def transform(self, X_train=None, X_valid=None, X_test=None, y_train=None):
        return {
            "X_train": self.process(X_train),
            "X_valid": self.process(X_valid),
            "X_test": self.process(X_test),
            "y_train": y_train
        }
