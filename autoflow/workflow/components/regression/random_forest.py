from autoflow.workflow.components.base import AutoFlowIterComponent
from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm


class RandomForestRegressor(
    AutoFlowIterComponent, AutoFlowRegressionAlgorithm,
):
    class__ = "RandomForestRegressor"
    module__ = "sklearn.ensemble"