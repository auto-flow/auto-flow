from hyperflow.pipeline.components.regression_base import HyperFlowRegressionAlgorithm


class ExtraTreesRegressor(
    HyperFlowRegressionAlgorithm,
):
    module__ = "sklearn.ensemble"
    class__ = "ETR"
