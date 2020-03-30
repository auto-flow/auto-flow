from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__ = ["DropAll"]


class DropAll(AutoPLFeatureEngineerAlgorithm):
    class__ = "DropAll"
    module__ = "autopipeline.feature_engineer.operate.drop_all"
