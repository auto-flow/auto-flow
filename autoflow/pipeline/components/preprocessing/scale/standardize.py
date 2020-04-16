from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__=["StandardScaler"]

class StandardScaler(AutoFlowFeatureEngineerAlgorithm):
    class__ = "StandardScaler"
    module__ = "sklearn.preprocessing"
