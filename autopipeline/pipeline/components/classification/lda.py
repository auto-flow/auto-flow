from copy import deepcopy
from typing import Dict

from autopipeline.pipeline.components.classification_base import AutoPLClassificationAlgorithm

__all__=["LDA"]

class LDA(AutoPLClassificationAlgorithm):
    class__ = "LinearDiscriminantAnalysis"
    module__ = "sklearn.discriminant_analysis"
    OVR__ = True


