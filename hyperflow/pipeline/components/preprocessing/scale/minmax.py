from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm

__all__=["MinMaxScaler"]

class MinMaxScaler(HyperFlowFeatureEngineerAlgorithm):
    class__ = "MinMaxScaler"
    module__ = "sklearn.preprocessing"


