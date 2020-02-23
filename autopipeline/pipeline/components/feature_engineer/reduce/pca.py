from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm

__all__=["PCA"]

class PCA(AutoPLPreprocessingAlgorithm):
    class__ = "PCA"
    module__ = "sklearn.decomposition"



