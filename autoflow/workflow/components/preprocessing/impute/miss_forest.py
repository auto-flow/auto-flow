from autoflow.workflow.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["MissForest"]


class MissForest(AutoFlowFeatureEngineerAlgorithm):
    class__ = "MissForest"
    module__ = "skimpute"
