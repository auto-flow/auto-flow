
from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm

__all__=["KernelPCA"]

class KernelPCA(AutoPLPreprocessingAlgorithm):
    class__ = "KernelPCA"
    module__ = "sklearn.decomposition"

