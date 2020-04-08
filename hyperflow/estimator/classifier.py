from hyperflow import HyperFlowEstimator


class HyperFlowClassifier(HyperFlowEstimator):
    checked_mainTask = "classification"

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)