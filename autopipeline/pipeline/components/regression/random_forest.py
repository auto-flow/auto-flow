from autopipeline.pipeline.components.regression_base import AutoPLRegressionAlgorithm


class RandomForest(
    AutoPLRegressionAlgorithm,
):
    class__ = "RandomForestRegressor"
    module__ = "sklearn.ensemble"