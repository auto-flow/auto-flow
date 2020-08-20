from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm

__all__ = ["SVR"]


class SVR(AutoFlowRegressionAlgorithm):
    class__ = "SVR"
    module__ = "sklearn.svm"
