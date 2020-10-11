from autoflow.workflow.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["PowerTransformer"]


class PowerTransformer(AutoFlowFeatureEngineerAlgorithm):
    class__ = "PowerTransformer"
    module__ = "sklearn.preprocessing"
