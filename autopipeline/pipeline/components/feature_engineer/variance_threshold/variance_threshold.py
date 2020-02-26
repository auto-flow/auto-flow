from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__=["VarianceThreshold"]

class VarianceThreshold(AutoPLPreprocessingAlgorithm):
    class__ = "sklearn.feature_selection"
    module__ = "VarianceThreshold"