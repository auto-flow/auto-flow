from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm


class RandomTreesEmbedding(AutoPLPreprocessingAlgorithm):
    module__ = "sklearn.ensemble"
    class__ = "RandomTreesEmbedding"
