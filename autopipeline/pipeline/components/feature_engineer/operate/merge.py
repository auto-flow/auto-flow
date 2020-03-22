from typing import Optional

from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm
from autopipeline.pipeline.dataframe import GenericDataFrame

__all__ = ["Merge"]


class Merge(AutoPLPreprocessingAlgorithm):

    def fit(self, X_train: GenericDataFrame, y_train=None,
            X_valid=None, y_valid=None,
            X_test=None, y_test=None):
        return self

    def process(self, X_origin: Optional[GenericDataFrame]) -> Optional[GenericDataFrame]:

        if X_origin is None:
            return None
        if isinstance(self.out_feat_grp, (list, tuple)):
            assert len(self.out_feat_grp) == 1
            self.out_feat_grp = self.out_feat_grp[0]
        # fixme: deepcopy
        X_origin.set_feat_grp(X_origin.feat_grp.replace(self.in_feat_grp, self.out_feat_grp))
        return X_origin

    def transform(self, X_train=None, X_valid=None, X_test=None, is_train=False):
        return {
            "X_train": self.process(X_train),
            "X_valid": self.process(X_valid),
            "X_test": self.process(X_test),
        }
