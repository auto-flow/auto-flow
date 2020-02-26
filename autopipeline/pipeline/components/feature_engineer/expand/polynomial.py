from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__ = ["PolynomialFeatures"]


class PolynomialFeatures(AutoPLPreprocessingAlgorithm):
    module__ = "sklearn.preprocessing"
    class__ = "PolynomialFeatures"
