from autoflow.pipeline.components.classification_base import AutoFlowClassificationAlgorithm

__all__=["LibSVM_SVC"]


class LibSVM_SVC(AutoFlowClassificationAlgorithm):
    class__ = "SVC"
    module__ = "sklearn.svm"
