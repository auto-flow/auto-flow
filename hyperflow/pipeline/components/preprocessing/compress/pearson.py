from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm

__all__ = ["Pearson"]

class Pearson(HyperFlowFeatureEngineerAlgorithm):
    class__ = "Pearson"
    module__ = "hyperflow.feature_engineer.compress.pearson"
    store_intermediate = True
    suspend_other_processes = True