from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm

__all__=["RandomKitchenSinks"]

class RandomKitchenSinks(HyperFlowFeatureEngineerAlgorithm):
    module__ = "sklearn.kernel_approximation"
    class__ = "RBFSampler"