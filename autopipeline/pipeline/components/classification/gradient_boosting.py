from copy import deepcopy

from autopipeline.pipeline.components.classification_base import AutoPLClassificationAlgorithm

__all__=["GradientBoostingClassifier"]


class GradientBoostingClassifier(AutoPLClassificationAlgorithm):
    module__ =  "sklearn.ensemble"
    class__ = "GradientBoostingClassifier"


