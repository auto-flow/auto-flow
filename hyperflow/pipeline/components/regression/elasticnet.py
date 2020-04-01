from hyperflow.pipeline.components.regression_base import HyperFlowRegressionAlgorithm


class ElasticNet(HyperFlowRegressionAlgorithm):
    class__ = "ElasticNet"
    module__ = "sklearn.linear_model"

