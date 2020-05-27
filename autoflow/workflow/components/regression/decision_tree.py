
from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm

__all__ = ["DecisionTree"]

class DecisionTree(AutoFlowRegressionAlgorithm):
    module__ = "sklearn.tree"
    class__ = "DecisionTreeRegressor"
