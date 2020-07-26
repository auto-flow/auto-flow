from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm

__all__ = ["SVC"]


class SVC(AutoFlowClassificationAlgorithm):
    class__ = "SVC"
    module__ = "sklearn.svm"
