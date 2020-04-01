from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm

__all__ = ["PolynomialFeatures"]


class PolynomialFeatures(HyperFlowFeatureEngineerAlgorithm):
    module__ = "sklearn.preprocessing"
    class__ = "PolynomialFeatures"
