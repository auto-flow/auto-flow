from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm

__all__=["PCA"]

class PCA(HyperFlowFeatureEngineerAlgorithm):
    class__ = "PCA"
    module__ = "sklearn.decomposition"



