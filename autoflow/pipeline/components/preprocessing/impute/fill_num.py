from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["FillNum"]


class FillNum(AutoFlowFeatureEngineerAlgorithm):
    class__ = "SimpleImputer"
    module__ = "sklearn.impute"
