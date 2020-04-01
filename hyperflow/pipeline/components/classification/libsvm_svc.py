from hyperflow.pipeline.components.classification_base import HyperFlowClassificationAlgorithm

__all__=["LibSVM_SVC"]


class LibSVM_SVC(HyperFlowClassificationAlgorithm):
    class__ = "SVC"
    module__ = "sklearn.svm"
