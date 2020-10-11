from autoflow.workflow.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__=["MinMaxScaler"]

class MinMaxScaler(AutoFlowFeatureEngineerAlgorithm):
    class__ = "MinMaxScaler"
    module__ = "sklearn.preprocessing"


