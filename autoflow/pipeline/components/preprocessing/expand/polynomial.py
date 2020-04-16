from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["PolynomialFeatures"]


class PolynomialFeatures(AutoFlowFeatureEngineerAlgorithm):
    module__ = "sklearn.preprocessing"
    class__ = "PolynomialFeatures"
