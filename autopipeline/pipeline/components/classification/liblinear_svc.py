from sklearn.calibration import CalibratedClassifierCV

from autopipeline.pipeline.components.base import AutoPLClassificationAlgorithm
from autopipeline.utils.data import softmax


class LibLinear_SVC(AutoPLClassificationAlgorithm):
    class__ = "LinearSVC"
    module__ = "sklearn.svm"
    OVR__ = True

    def predict_proba(self, X):
        decision_function=self.estimator.decision_function(X)
        return softmax(decision_function)