from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__=["RandomKitchenSinks"]

class RandomKitchenSinks(AutoPLPreprocessingAlgorithm):
    module__ = "sklearn.kernel_approximation"
    class__ = "RBFSampler"