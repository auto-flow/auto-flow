from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__=["VarianceThreshold"]

class VarianceThreshold(AutoPLPreprocessingAlgorithm):
    class__ = "VarianceThreshold"
    module__ = "sklearn.feature_selection"