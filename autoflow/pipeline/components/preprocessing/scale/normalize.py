from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__=["NormalizerComponent"]

class NormalizerComponent(AutoFlowFeatureEngineerAlgorithm):
    class__ = "Normalizer"
    module__ = "sklearn.preprocessing"