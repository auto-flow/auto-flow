from autoflow.pipeline.components.regression_base import AutoFlowRegressionAlgorithm


class RandomForest(
    AutoFlowRegressionAlgorithm,
):
    class__ = "RandomForestRegressor"
    module__ = "sklearn.ensemble"