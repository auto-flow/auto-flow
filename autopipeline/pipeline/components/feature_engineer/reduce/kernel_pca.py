
from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm


class KernelPCA(AutoPLPreprocessingAlgorithm):
    class__ = "KernelPCA"
    module__ = "sklearn.decomposition"

