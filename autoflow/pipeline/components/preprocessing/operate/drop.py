from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["DropAll"]


class DropAll(AutoFlowFeatureEngineerAlgorithm):
    class__ = "DropAll"
    module__ = "autoflow.feature_engineer.operate.drop_all"
