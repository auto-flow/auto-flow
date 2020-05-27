from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm

__all__ = ["ElasticNet"]

class ElasticNet(AutoFlowRegressionAlgorithm):
    class__ = "ElasticNet"
    module__ = "sklearn.linear_model"

