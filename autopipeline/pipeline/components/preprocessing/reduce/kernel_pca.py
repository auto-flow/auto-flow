from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__=["KernelPCA"]

class KernelPCA(AutoPLFeatureEngineerAlgorithm):
    class__ = "KernelPCA"
    module__ = "sklearn.decomposition"

