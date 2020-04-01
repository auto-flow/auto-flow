from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm

__all__ = ["Variance"]

class Variance(HyperFlowFeatureEngineerAlgorithm):
    class__ = "Variance"
    module__ = "hyperflow.feature_engineer.compress.variance"