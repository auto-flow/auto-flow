from typing import Dict

from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__=["FeatureAgglomeration"]

class FeatureAgglomeration(AutoPLPreprocessingAlgorithm):
    module__ = "sklearn.cluster"
    class__ = "FeatureAgglomeration"
