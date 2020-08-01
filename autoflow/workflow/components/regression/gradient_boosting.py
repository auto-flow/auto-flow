from autoflow.workflow.components.iter_algo import AutoFlowIterComponent
from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm

__all__ = ["GradientBoostingRegressor"]


class GradientBoostingRegressor(AutoFlowIterComponent, AutoFlowRegressionAlgorithm):
    class__ = "GradientBoostingRegressor"
    module__ = "sklearn.ensemble"
