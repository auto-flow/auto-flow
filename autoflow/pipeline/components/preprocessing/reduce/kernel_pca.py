from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__=["KernelPCA"]

class KernelPCA(AutoFlowFeatureEngineerAlgorithm):
    class__ = "KernelPCA"
    module__ = "sklearn.decomposition"

