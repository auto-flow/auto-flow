from hyperflow.pipeline.components.classification_base import HyperFlowClassificationAlgorithm

__all__=["DecisionTree"]


class DecisionTree(HyperFlowClassificationAlgorithm):
    class__ = "DecisionTreeClassifier"
    module__ = "sklearn.tree"
