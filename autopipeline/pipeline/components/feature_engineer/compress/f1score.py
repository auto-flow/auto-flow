from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__ = ["F1Score"]

class F1Score(AutoPLPreprocessingAlgorithm):
    class__ = "F1Score"
    module__ = "autopipeline.feature_engineer.compress.variance"