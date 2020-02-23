from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm


class NormalizerComponent(AutoPLPreprocessingAlgorithm):
    class__ = "Normalizer"
    module__ = "sklearn.preprocessing"