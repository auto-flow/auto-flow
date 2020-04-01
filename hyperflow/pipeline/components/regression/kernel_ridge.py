from hyperflow.pipeline.components.regression_base import HyperFlowRegressionAlgorithm


class ElasticNet(HyperFlowRegressionAlgorithm):
    class__ = "KernelRidge"
    module__ = "sklearn.kernel_ridge"

