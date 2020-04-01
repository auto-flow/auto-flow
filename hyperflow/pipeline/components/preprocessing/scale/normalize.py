from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm

__all__=["NormalizerComponent"]

class NormalizerComponent(HyperFlowFeatureEngineerAlgorithm):
    class__ = "Normalizer"
    module__ = "sklearn.preprocessing"