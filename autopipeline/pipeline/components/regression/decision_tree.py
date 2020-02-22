
from autopipeline.pipeline.components.base import AutoPLRegressionAlgorithm


class DecisionTree(AutoPLRegressionAlgorithm):
    module__ = "sklearn.tree"
    class__ = "DecisionTreeRegressor"
