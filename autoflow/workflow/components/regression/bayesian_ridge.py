from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm


__all__ = ["BayesianRidge"]


class BayesianRidge(AutoFlowRegressionAlgorithm):
    class__ = "BayesianRidge"
    module__ = "sklearn.linear_model"

