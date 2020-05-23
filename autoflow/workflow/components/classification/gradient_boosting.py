from copy import deepcopy

from autoflow.workflow.components.base import AutoFlowIterComponent
from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm

__all__=["GradientBoostingClassifier"]


class GradientBoostingClassifier(AutoFlowIterComponent, AutoFlowClassificationAlgorithm):
    module__ =  "sklearn.ensemble"
    class__ = "GradientBoostingClassifier"


