from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__=["RandomTreesEmbedding"]

class RandomTreesEmbedding(AutoPLFeatureEngineerAlgorithm):
    module__ = "sklearn.ensemble"
    class__ = "RandomTreesEmbedding"
