
from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["F1Score"]


class F1Score(AutoFlowFeatureEngineerAlgorithm):
    class__ = "F1Score"
    module__ = "autoflow.feature_engineer.compress.f1score"
    store_intermediate = True
    suspend_other_processes = True