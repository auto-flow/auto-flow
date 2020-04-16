from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["Pearson"]

class Pearson(AutoFlowFeatureEngineerAlgorithm):
    class__ = "Pearson"
    module__ = "autoflow.feature_engineer.compress.pearson"
    store_intermediate = True
    suspend_other_processes = True