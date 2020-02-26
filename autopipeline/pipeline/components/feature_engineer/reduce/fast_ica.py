from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__=["FastICA"]

class FastICA(AutoPLPreprocessingAlgorithm):
    class__ = "FastICA"
    module__ = "sklearn.decomposition"
