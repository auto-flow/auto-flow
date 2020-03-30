from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__=["NormalizerComponent"]

class NormalizerComponent(AutoPLFeatureEngineerAlgorithm):
    class__ = "Normalizer"
    module__ = "sklearn.preprocessing"