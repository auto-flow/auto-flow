from autopipeline.pipeline.components.base import AutoPLRegressionAlgorithm


class ExtraTreesRegressor(
    AutoPLRegressionAlgorithm,
):
    module__ = "sklearn.ensemble"
    class__ = "ETR"
