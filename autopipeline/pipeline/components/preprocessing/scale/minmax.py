from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__=["MinMaxScaler"]

class MinMaxScaler(AutoPLFeatureEngineerAlgorithm):
    class__ = "MinMaxScaler"
    module__ = "sklearn.preprocessing"


