from autopipeline.pipeline.components.base import AutoPLClassificationAlgorithm


class LibSVM_SVC(AutoPLClassificationAlgorithm):
    class__ = "SVC"
    module__ = "sklearn.svm"
