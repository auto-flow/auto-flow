from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__ = ["QuantileTransformer"]


class QuantileTransformer(AutoPLPreprocessingAlgorithm):
    class__ = "QuantileTransformer"
    module__ = "sklearn.preprocessing"
