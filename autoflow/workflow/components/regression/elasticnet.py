from autoflow.workflow.components.base import AutoFlowIterComponent
from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm

__all__ = ["ElasticNet"]

class ElasticNet(AutoFlowIterComponent, AutoFlowRegressionAlgorithm):
    class__ = "ElasticNet"
    module__ = "sklearn.linear_model"
    iterations_name = "max_iter"

