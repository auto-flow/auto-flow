from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm

__all__=["QuantileTransformerComponent"]

class QuantileTransformerComponent(AutoPLPreprocessingAlgorithm):
    class__ = "QuantileTransformer"
    module__ = "sklearn.preprocessing"
