from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm


class ExtraTreesRegressor(
    AutoFlowRegressionAlgorithm,
):
    module__ = "sklearn.ensemble"
    class__ = "ETR"
