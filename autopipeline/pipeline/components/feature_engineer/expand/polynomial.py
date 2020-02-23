from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm

__all__=["PolynomialFeatures"]

class PolynomialFeatures(AutoPLPreprocessingAlgorithm):
    module__ = "sklearn.preprocessing"
    class__ = "PolynomialFeatures"
