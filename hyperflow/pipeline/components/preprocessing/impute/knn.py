from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm

__all__ = ["KNNImputer"]


class KNNImputer(HyperFlowFeatureEngineerAlgorithm):
    class__ = "KNNImputer"
    module__ = "sklearn.impute"
