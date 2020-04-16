from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["KNNImputer"]


class KNNImputer(AutoFlowFeatureEngineerAlgorithm):
    class__ = "KNNImputer"
    module__ = "sklearn.impute"
