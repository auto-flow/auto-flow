from copy import deepcopy
from typing import Dict

from hyperflow.pipeline.components.classification_base import HyperFlowClassificationAlgorithm

__all__=["LDA"]

class LDA(HyperFlowClassificationAlgorithm):
    class__ = "LinearDiscriminantAnalysis"
    module__ = "sklearn.discriminant_analysis"
    OVR__ = True


