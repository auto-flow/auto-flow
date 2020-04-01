
from hyperflow.pipeline.components.regression_base import HyperFlowRegressionAlgorithm


class DecisionTree(HyperFlowRegressionAlgorithm):
    module__ = "sklearn.tree"
    class__ = "DecisionTreeRegressor"
