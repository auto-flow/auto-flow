from autopipeline.pipeline.components.base import AutoPLRegressionAlgorithm


class SGD(
    AutoPLRegressionAlgorithm,
):
    module__ = "sklearn.linear_model.stochastic_gradient"
    class__ = "SGDRegressor"