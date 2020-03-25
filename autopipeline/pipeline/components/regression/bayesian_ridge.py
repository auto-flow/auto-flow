from autopipeline.pipeline.components.regression_base import AutoPLRegressionAlgorithm


class BayesianRidge(AutoPLRegressionAlgorithm):
    class__ = "BayesianRidge"
    module__ = "sklearn.linear_model"

