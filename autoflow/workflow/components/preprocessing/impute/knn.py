from autoflow.workflow.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["KNNImputer"]


class KNNImputer(AutoFlowFeatureEngineerAlgorithm):
    class__ = "GBTImputer"
    module__ = "autoflow.feature_engineer.impute"
