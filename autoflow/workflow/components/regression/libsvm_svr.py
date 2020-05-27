from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm

__all__ = ["LibSVM_SVR"]


class LibSVM_SVR(AutoFlowRegressionAlgorithm):
    class__ = "SVR"
    module__ = "sklearn.svm"
