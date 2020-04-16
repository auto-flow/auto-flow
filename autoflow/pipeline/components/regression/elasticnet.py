from autoflow.pipeline.components.regression_base import AutoFlowRegressionAlgorithm


class ElasticNet(AutoFlowRegressionAlgorithm):
    class__ = "ElasticNet"
    module__ = "sklearn.linear_model"

