from sklearn.calibration import CalibratedClassifierCV

from hyperflow.pipeline.components.classification_base import HyperFlowClassificationAlgorithm
from hyperflow.utils.data import softmax

__all__=["LibLinear_SVC"]

class LibLinear_SVC(HyperFlowClassificationAlgorithm):
    class__ = "LinearSVC"
    module__ = "sklearn.svm"
    OVR__ = True

    def predict_proba(self, X):
        decision_function=self.estimator.decision_function(X)
        return softmax(decision_function)