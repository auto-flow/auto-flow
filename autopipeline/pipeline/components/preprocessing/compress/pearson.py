from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__ = ["Pearson"]

class Pearson(AutoPLFeatureEngineerAlgorithm):
    class__ = "Pearson"
    module__ = "autopipeline.feature_engineer.compress.variance"