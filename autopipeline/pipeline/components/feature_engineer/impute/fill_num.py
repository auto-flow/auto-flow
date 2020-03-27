from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__ = ["FillNum"]


class FillNum(AutoPLPreprocessingAlgorithm):
    class__ = "SimpleImputer"
    module__ = "sklearn.impute"
