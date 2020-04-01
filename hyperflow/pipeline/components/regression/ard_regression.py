from hyperflow.pipeline.components.regression_base import HyperFlowRegressionAlgorithm


class ARDRegression(HyperFlowRegressionAlgorithm):
    class__ = "ARDRegression"
    module__ = "sklearn.linear_model"

