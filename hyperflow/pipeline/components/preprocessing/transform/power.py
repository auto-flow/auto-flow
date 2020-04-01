from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm

__all__ = ["PowerTransformer"]


class PowerTransformer(HyperFlowFeatureEngineerAlgorithm):
    class__ = "PowerTransformer"
    module__ = "sklearn.preprocessing"
