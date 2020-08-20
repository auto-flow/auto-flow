import numpy as np

from autoflow.workflow.components.iter_algo import LgbmIterativeMixIn
from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm

__all__ = ["LGBMRegressor"]


class LGBMRegressor(LgbmIterativeMixIn, AutoFlowRegressionAlgorithm):
    class__ = "LGBMRegressor"
    module__ = "autoflow.estimator.wrap_lightgbm"

    boost_model = True
    tree_model = True
    support_early_stopping = True

