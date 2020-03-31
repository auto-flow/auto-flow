from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__ = ["KNNImputer"]


class KNNImputer(AutoPLFeatureEngineerAlgorithm):
    class__ = "KNNImputer"
    module__ = "sklearn.impute"
