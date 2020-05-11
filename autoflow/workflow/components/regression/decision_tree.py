
from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm


class DecisionTree(AutoFlowRegressionAlgorithm):
    module__ = "sklearn.tree"
    class__ = "DecisionTreeRegressor"
