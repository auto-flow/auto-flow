from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__=["PCA"]

class PCA(AutoPLFeatureEngineerAlgorithm):
    class__ = "PCA"
    module__ = "sklearn.decomposition"



