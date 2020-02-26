from copy import deepcopy
from typing import Dict

from autopipeline.pipeline.components.base import AutoPLComponent


class AutoPLPreprocessingAlgorithm(AutoPLComponent):
    """Provide an abstract interface for preprocessing algorithms in
    auto-sklearn.

    See :ref:`extending` for more information."""
    def before_trans_X(self, X):
        return X

    def after_trans_X(self,X):
        return X

    def transform(self, X):
        if not self.estimator or (not hasattr(self.estimator, "transform")):
            raise NotImplementedError()
        X=self.before_trans_X(X)
        return self.after_trans_X(self.estimator.transform(X))

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

