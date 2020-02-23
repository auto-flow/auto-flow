from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm


class RandomKitchenSinks(AutoPLPreprocessingAlgorithm):
    module__ = "sklearn.kernel_approximation"
    class__ = "RBFSampler"