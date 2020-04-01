from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm

__all__=["RandomTreesEmbedding"]

class RandomTreesEmbedding(HyperFlowFeatureEngineerAlgorithm):
    module__ = "sklearn.ensemble"
    class__ = "RandomTreesEmbedding"
