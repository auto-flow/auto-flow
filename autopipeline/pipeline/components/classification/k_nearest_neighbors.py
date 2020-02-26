from autopipeline.pipeline.components.classification_base import AutoPLClassificationAlgorithm

__all__=["KNearestNeighborsClassifier"]


class KNearestNeighborsClassifier(AutoPLClassificationAlgorithm):
    class__ = "KNeighborsClassifier"
    module__ = "sklearn.neighbors"
    # fixme ovr