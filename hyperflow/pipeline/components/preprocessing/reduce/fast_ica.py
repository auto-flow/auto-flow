from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm

__all__=["FastICA"]

class FastICA(HyperFlowFeatureEngineerAlgorithm):
    class__ = "FastICA"
    module__ = "sklearn.decomposition"
