from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm


class ElasticNet(AutoFlowRegressionAlgorithm):
    class__ = "KernelRidge"
    module__ = "sklearn.kernel_ridge"

