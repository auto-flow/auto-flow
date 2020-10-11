
from autoflow.workflow.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["F1Score"]


class F1Score(AutoFlowFeatureEngineerAlgorithm):
    class__ = "F1Score"
    module__ = "autoflow.feature_engineer.compress.f1score"
    cache_intermediate = True
    suspend_other_processes = True