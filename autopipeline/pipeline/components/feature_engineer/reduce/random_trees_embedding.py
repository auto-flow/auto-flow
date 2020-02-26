from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__=["RandomTreesEmbedding"]

class RandomTreesEmbedding(AutoPLPreprocessingAlgorithm):
    module__ = "sklearn.ensemble"
    class__ = "RandomTreesEmbedding"
