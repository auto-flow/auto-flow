from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm


class PCA(AutoPLPreprocessingAlgorithm):
    class__ = "PCA"
    module__ = "sklearn.decomposition"



