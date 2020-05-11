import pandas as pd

from autoflow.workflow.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm


class BaseEncoder(AutoFlowFeatureEngineerAlgorithm):

    def _transform_procedure(self, X):
        if X is None:
            return None
        else:
            X_ = X.astype(str)
            trans = self.estimator.transform(X_)
            return trans

    def core_fit(self, estimator, X, y, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None, columns_metadata=None):
        X_ = X.astype(str)
        return estimator.fit(X_, y)
