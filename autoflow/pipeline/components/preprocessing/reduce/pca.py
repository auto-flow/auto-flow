from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__=["PCA"]

class PCA(AutoFlowFeatureEngineerAlgorithm):
    class__ = "PCA"
    module__ = "sklearn.decomposition"



