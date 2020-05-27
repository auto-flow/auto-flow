from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm

__all__ = ["KernelRidge"]


class KernelRidge(AutoFlowRegressionAlgorithm):
    class__ = "KernelRidge"
    module__ = "sklearn.kernel_ridge"
