from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__ = ["Variance"]

class Variance(AutoPLFeatureEngineerAlgorithm):
    class__ = "Variance"
    module__ = "autopipeline.feature_engineer.compress.variance"