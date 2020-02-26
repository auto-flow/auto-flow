from autopipeline.pipeline.components.regression_base import AutoPLRegressionAlgorithm


class ARDRegression(AutoPLRegressionAlgorithm):
    class__ = "ARDRegression"
    module__ = "sklearn.linear_model"

