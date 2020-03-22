from autopipeline.pipeline.components.regression_base import AutoPLRegressionAlgorithm


class ElasticNet(AutoPLRegressionAlgorithm):
    class__ = "KernelRidge"
    module__ = "sklearn.kernel_ridge"

