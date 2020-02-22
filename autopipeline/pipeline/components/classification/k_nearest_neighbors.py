
from autopipeline.pipeline.components.base import AutoPLClassificationAlgorithm


class KNearestNeighborsClassifier(AutoPLClassificationAlgorithm):
    class__ = "KNeighborsClassifier"
    module__ = "sklearn.neighbors"
    # fixme ovr