from autoflow.workflow.components.iter_algo import AutoFlowIterComponent
from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm

__all__ = ["ExtraTreesRegressor"]


class ExtraTreesRegressor(
    AutoFlowIterComponent, AutoFlowRegressionAlgorithm,
):
    module__ = "sklearn.ensemble"
    class__ = "ExtraTreesRegressor"
