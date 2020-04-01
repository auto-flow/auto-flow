from copy import deepcopy

from hyperflow.pipeline.components.classification_base import HyperFlowClassificationAlgorithm

__all__=["GradientBoostingClassifier"]


class GradientBoostingClassifier(HyperFlowClassificationAlgorithm):
    module__ =  "sklearn.ensemble"
    class__ = "GradientBoostingClassifier"


