from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__ = ["PowerTransformer"]


class PowerTransformer(AutoPLPreprocessingAlgorithm):
    class__ = "PowerTransformer"
    module__ = "sklearn.preprocessing"
