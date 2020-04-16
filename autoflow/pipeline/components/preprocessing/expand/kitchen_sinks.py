from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__=["RandomKitchenSinks"]

class RandomKitchenSinks(AutoFlowFeatureEngineerAlgorithm):
    module__ = "sklearn.kernel_approximation"
    class__ = "RBFSampler"