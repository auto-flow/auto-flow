from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm

__all__ = ["DropAll"]


class DropAll(HyperFlowFeatureEngineerAlgorithm):
    class__ = "DropAll"
    module__ = "hyperflow.feature_engineer.operate.drop_all"
