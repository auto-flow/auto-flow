from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm
from autoflow.workflow.components.iter_algo import LgbmIterativeMixIn

__all__ = ["LGBMClassifier"]


class LGBMClassifier(LgbmIterativeMixIn, AutoFlowClassificationAlgorithm):
    class__ = "LGBMClassifier"
    module__ = "autoflow.estimator.wrap_lightgbm"

    boost_model = True
    tree_model = True
    support_early_stopping = True
