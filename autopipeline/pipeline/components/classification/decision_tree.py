from autopipeline.pipeline.components.classification_base import AutoPLClassificationAlgorithm

__all__=["DecisionTree"]


class DecisionTree(AutoPLClassificationAlgorithm):
    class__ = "DecisionTreeClassifier"
    module__ = "sklearn.tree"
