from copy import deepcopy

from hyperflow.pipeline.components.regression_base import HyperFlowRegressionAlgorithm


class GradientBoosting(HyperFlowRegressionAlgorithm):
    class__ = "GradientBoostingRegressor"
    module__ = "sklearn.ensemble"


