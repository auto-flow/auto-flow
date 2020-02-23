from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm

__all__=["NormalizerComponent"]

class NormalizerComponent(AutoPLPreprocessingAlgorithm):
    class__ = "Normalizer"
    module__ = "sklearn.preprocessing"