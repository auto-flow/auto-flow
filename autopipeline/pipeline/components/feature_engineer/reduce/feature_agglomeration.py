from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm

__all__=["FeatureAgglomeration"]

class FeatureAgglomeration(AutoPLPreprocessingAlgorithm):
    module__ = "sklearn.cluster"
    class__ = "FeatureAgglomeration"