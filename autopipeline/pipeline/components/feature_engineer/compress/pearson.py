from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__ = ["Pearson"]

class Pearson(AutoPLPreprocessingAlgorithm):
    class__ = "Pearson"
    module__ = "autopipeline.pipeline.proxy.pearson"