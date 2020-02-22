
from autopipeline.pipeline.components.base import AutoPLClassificationAlgorithm


class DecisionTree(AutoPLClassificationAlgorithm):
    class__ = "DecisionTreeClassifier"
    module__ = "sklearn.tree"
