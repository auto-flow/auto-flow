from autopipeline.pipeline.components.base import AutoPLRegressionAlgorithm


class RidgeRegression(AutoPLRegressionAlgorithm):
    module__ = "sklearn.linear_model"
    class__ = "Ridge"