from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__ = ["Variance"]

class Variance(AutoPLPreprocessingAlgorithm):
    class__ = "Pearson"
    module__ = "autopipeline.feature_engineer.compress.variance"