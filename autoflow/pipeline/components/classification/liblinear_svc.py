from sklearn.calibration import CalibratedClassifierCV

from autoflow.pipeline.components.classification_base import AutoFlowClassificationAlgorithm
from autoflow.utils.data import softmax

__all__=["LibLinear_SVC"]

class LibLinear_SVC(AutoFlowClassificationAlgorithm):
    class__ = "LinearSVC"
    module__ = "sklearn.svm"
    OVR__ = True
