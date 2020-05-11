from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm


class ARDRegression(AutoFlowRegressionAlgorithm):
    class__ = "ARDRegression"
    module__ = "sklearn.linear_model"

