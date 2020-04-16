from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["Variance"]

class Variance(AutoFlowFeatureEngineerAlgorithm):
    class__ = "Variance"
    module__ = "autoflow.feature_engineer.compress.variance"