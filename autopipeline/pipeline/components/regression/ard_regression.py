from autopipeline.pipeline.components.base import AutoPLRegressionAlgorithm


class ARDRegression(AutoPLRegressionAlgorithm):
    class__ = "ARDRegression"
    module__ = "sklearn.linear_model"

