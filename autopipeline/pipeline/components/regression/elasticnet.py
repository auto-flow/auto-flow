from autopipeline.pipeline.components.regression_base import AutoPLRegressionAlgorithm


class ElasticNet(AutoPLRegressionAlgorithm):
    class__ = "ElasticNet"
    module__ = "sklearn.linear_model"

