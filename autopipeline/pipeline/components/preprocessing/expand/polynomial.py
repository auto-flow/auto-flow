from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__ = ["PolynomialFeatures"]


class PolynomialFeatures(AutoPLFeatureEngineerAlgorithm):
    module__ = "sklearn.preprocessing"
    class__ = "PolynomialFeatures"
