from autopipeline.pipeline.components.classification_base import AutoPLClassificationAlgorithm

__all__=["LibSVM_SVC"]


class LibSVM_SVC(AutoPLClassificationAlgorithm):
    class__ = "SVC"
    module__ = "sklearn.svm"
