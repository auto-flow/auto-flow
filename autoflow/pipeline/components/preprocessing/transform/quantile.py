from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["QuantileTransformer"]


class QuantileTransformer(AutoFlowFeatureEngineerAlgorithm):
    class__ = "QuantileTransformer"
    module__ = "sklearn.preprocessing"
