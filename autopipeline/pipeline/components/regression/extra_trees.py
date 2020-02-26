from autopipeline.pipeline.components.regression_base import AutoPLRegressionAlgorithm


class ExtraTreesRegressor(
    AutoPLRegressionAlgorithm,
):
    module__ = "sklearn.ensemble"
    class__ = "ETR"
