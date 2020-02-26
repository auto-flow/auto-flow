from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__=["NormalizerComponent"]

class NormalizerComponent(AutoPLPreprocessingAlgorithm):
    class__ = "Normalizer"
    module__ = "sklearn.preprocessing"