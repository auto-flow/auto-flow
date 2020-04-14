import pandas as pd

from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm


class BaseEncoder(HyperFlowFeatureEngineerAlgorithm):

    def _transform_proc(self, X):
        if X is None:
            return None
        else:
            trans = self.estimator.transform(X)
            trans[pd.DataFrame(X) == -999] = -999  # todo: 有没有更优化的解决办法
            return trans

    def core_fit(self, estimator, X, y, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None, columns_metadata=None):
        X_ = X.astype(str)
        return estimator.fit(X_, y)
