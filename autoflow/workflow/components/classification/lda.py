from copy import deepcopy
from typing import Dict

from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm

__all__=["LinearDiscriminantAnalysis"]

class LinearDiscriminantAnalysis(AutoFlowClassificationAlgorithm):
    class__ = "LinearDiscriminantAnalysis"
    module__ = "sklearn.discriminant_analysis"
    OVR__ = True


