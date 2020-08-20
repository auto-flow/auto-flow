from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm
from autoflow.workflow.components.iter_algo import LgbmIterativeMixIn

__all__ = ["GBTLRClassifier"]


class GBTLRClassifier(LgbmIterativeMixIn, AutoFlowClassificationAlgorithm):
    class__ = "GBTLRClassifier"
    module__ = "autoflow.estimator.gbt_lr"

    boost_model = True
    tree_model = True
    support_early_stopping = True

