from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm


class VarianceThreshold(AutoPLPreprocessingAlgorithm):
    class__ = "sklearn.feature_selection"
    module__ = "VarianceThreshold"