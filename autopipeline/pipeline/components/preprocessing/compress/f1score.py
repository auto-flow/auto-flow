from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__ = ["F1Score"]

class F1Score(AutoPLFeatureEngineerAlgorithm):
    class__ = "F1Score"
    module__ = "autopipeline.feature_engineer.compress.variance"