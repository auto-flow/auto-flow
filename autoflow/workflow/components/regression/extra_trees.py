from autoflow.workflow.components.base import AutoFlowIterComponent
from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm


class ExtraTreesRegressor(
    AutoFlowIterComponent, AutoFlowRegressionAlgorithm,
):
    module__ = "sklearn.ensemble"
    class__ = "ExtraTreesRegressor"
