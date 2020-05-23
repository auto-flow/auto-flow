from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm

__all__ = ["LibLinear_SVC"]


class LibLinear_SVC(AutoFlowClassificationAlgorithm):
    class__ = "LinearSVC"
    module__ = "sklearn.svm"
    OVR__ = True
