from hyperflow.pipeline.components.regression_base import HyperFlowRegressionAlgorithm


class BayesianRidge(HyperFlowRegressionAlgorithm):
    class__ = "BayesianRidge"
    module__ = "sklearn.linear_model"

