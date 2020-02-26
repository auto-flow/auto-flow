from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__=["PCA"]

class PCA(AutoPLPreprocessingAlgorithm):
    class__ = "PCA"
    module__ = "sklearn.decomposition"



