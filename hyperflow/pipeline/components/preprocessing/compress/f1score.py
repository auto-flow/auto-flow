
from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm

__all__ = ["F1Score"]


class F1Score(HyperFlowFeatureEngineerAlgorithm):
    class__ = "F1Score"
    module__ = "hyperflow.feature_engineer.compress.f1score"
    store_intermediate = True