
from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm

__all__ = ["DecisionTreeRegressor"]

class DecisionTreeRegressor(AutoFlowRegressionAlgorithm):
    module__ = "sklearn.tree"
    class__ = "DecisionTreeRegressor"
