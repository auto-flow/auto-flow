from autopipeline.pipeline.components.base import AutoPLComponent
from autopipeline.utils.data import densify


class AutoPLRegressionAlgorithm(AutoPLComponent):
    """Provide an abstract interface for regression algorithms in
    auto-sklearn.

    Make a subclass of this and put it into the directory
    `autosklearn/pipeline/components/regression` to make it available."""


    def _pred_or_trans(self, X_train, X_valid=None, X_test=None):
        return self.estimator.predict(X_train)

    def predict(self, X):
        return self.pred_or_trans(X)
