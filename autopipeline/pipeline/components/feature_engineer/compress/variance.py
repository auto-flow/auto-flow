from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm


class Variance(AutoPLPreprocessingAlgorithm):
    class__ = "Pearson"
    module__ = "autopipeline.pipeline.proxy.variance"