from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__ = ["PowerTransformer"]


class PowerTransformer(AutoPLFeatureEngineerAlgorithm):
    class__ = "PowerTransformer"
    module__ = "sklearn.preprocessing"
