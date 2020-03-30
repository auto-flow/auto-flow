from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__=["RandomKitchenSinks"]

class RandomKitchenSinks(AutoPLFeatureEngineerAlgorithm):
    module__ = "sklearn.kernel_approximation"
    class__ = "RBFSampler"