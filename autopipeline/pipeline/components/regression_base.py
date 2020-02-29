from autopipeline.pipeline.components.base import AutoPLComponent
from autopipeline.utils.data import densify


class AutoPLRegressionAlgorithm(AutoPLComponent):
    """Provide an abstract interface for regression algorithms in
    auto-sklearn.

    Make a subclass of this and put it into the directory
    `autosklearn/pipeline/components/regression` to make it available."""


    def after_process_pred_y(self,y):
        return y

    def predict(self, X):
        X=densify(X)
        if not self.estimator:
            raise NotImplementedError()
        pred_y= self.estimator.predict(X)
        return self.after_process_pred_y(pred_y)

    def score(self,X,y):
        return self.estimator.score(X,y)