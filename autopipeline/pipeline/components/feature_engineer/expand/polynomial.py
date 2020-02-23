from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm


class PolynomialFeatures(AutoPLPreprocessingAlgorithm):
    module__ = "sklearn.preprocessing"
    class__ = "PolynomialFeatures"
