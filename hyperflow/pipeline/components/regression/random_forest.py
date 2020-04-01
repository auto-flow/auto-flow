from hyperflow.pipeline.components.regression_base import HyperFlowRegressionAlgorithm


class RandomForest(
    HyperFlowRegressionAlgorithm,
):
    class__ = "RandomForestRegressor"
    module__ = "sklearn.ensemble"