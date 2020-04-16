from copy import deepcopy
from typing import Dict

from autoflow.pipeline.components.classification_base import AutoFlowClassificationAlgorithm

__all__=["LDA"]

class LDA(AutoFlowClassificationAlgorithm):
    class__ = "LinearDiscriminantAnalysis"
    module__ = "sklearn.discriminant_analysis"
    OVR__ = True


