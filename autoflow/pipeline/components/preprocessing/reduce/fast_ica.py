from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__=["FastICA"]

class FastICA(AutoFlowFeatureEngineerAlgorithm):
    class__ = "FastICA"
    module__ = "sklearn.decomposition"
