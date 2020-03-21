from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm


class Pearson(AutoPLPreprocessingAlgorithm):
    class__ = "Pearson"
    module__ = "autopipeline.pipeline.proxy.pearson"