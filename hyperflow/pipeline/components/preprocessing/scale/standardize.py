from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm

__all__=["StandardScaler"]

class StandardScaler(HyperFlowFeatureEngineerAlgorithm):
    class__ = "StandardScaler"
    module__ = "sklearn.preprocessing"
