from autoflow.pipeline.components.classification_base import AutoFlowClassificationAlgorithm

__all__=["DecisionTree"]


class DecisionTree(AutoFlowClassificationAlgorithm):
    class__ = "DecisionTreeClassifier"
    module__ = "sklearn.tree"
