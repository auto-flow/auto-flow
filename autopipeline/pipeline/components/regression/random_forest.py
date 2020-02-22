from autopipeline.pipeline.components.base import AutoPLRegressionAlgorithm

class RandomForest(
    AutoPLRegressionAlgorithm,
):
    class__ = "RandomForestRegressor"
    module__ = "sklearn.ensemble"