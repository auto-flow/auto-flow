
from autopipeline.pipeline.components.regression_base import AutoPLRegressionAlgorithm


class DecisionTree(AutoPLRegressionAlgorithm):
    module__ = "sklearn.tree"
    class__ = "DecisionTreeRegressor"
