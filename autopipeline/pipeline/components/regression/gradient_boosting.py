from autopipeline.pipeline.components.base import AutoPLRegressionAlgorithm


class GradientBoosting(AutoPLRegressionAlgorithm):
    class__ = "GradientBoostingRegressor"
    module__ = "sklearn.ensemble"
