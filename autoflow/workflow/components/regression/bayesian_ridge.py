from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm


class BayesianRidge(AutoFlowRegressionAlgorithm):
    class__ = "BayesianRidge"
    module__ = "sklearn.linear_model"

