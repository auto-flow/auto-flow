from copy import deepcopy

from autoflow.pipeline.components.classification_base import AutoFlowClassificationAlgorithm

__all__=["GradientBoostingClassifier"]


class GradientBoostingClassifier(AutoFlowClassificationAlgorithm):
    module__ =  "sklearn.ensemble"
    class__ = "GradientBoostingClassifier"


