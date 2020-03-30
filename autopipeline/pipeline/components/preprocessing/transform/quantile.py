from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__ = ["QuantileTransformer"]


class QuantileTransformer(AutoPLFeatureEngineerAlgorithm):
    class__ = "QuantileTransformer"
    module__ = "sklearn.preprocessing"
