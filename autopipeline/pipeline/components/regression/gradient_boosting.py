from copy import deepcopy

from autopipeline.pipeline.components.regression_base import AutoPLRegressionAlgorithm


class GradientBoosting(AutoPLRegressionAlgorithm):
    class__ = "GradientBoostingRegressor"
    module__ = "sklearn.ensemble"


