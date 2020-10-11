from autoflow.workflow.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__=["RandomTreesEmbedding"]

class RandomTreesEmbedding(AutoFlowFeatureEngineerAlgorithm):
    module__ = "sklearn.ensemble"
    class__ = "RandomTreesEmbedding"
