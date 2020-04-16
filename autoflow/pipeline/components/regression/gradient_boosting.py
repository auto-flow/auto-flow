from copy import deepcopy

from autoflow.pipeline.components.regression_base import AutoFlowRegressionAlgorithm


class GradientBoosting(AutoFlowRegressionAlgorithm):
    class__ = "GradientBoostingRegressor"
    module__ = "sklearn.ensemble"


