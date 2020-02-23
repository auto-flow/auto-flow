from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm


class QuantileTransformerComponent(AutoPLPreprocessingAlgorithm):
    class__ = "QuantileTransformer"
    module__ = "sklearn.preprocessing"
