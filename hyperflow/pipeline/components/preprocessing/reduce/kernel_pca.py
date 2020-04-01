from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm

__all__=["KernelPCA"]

class KernelPCA(HyperFlowFeatureEngineerAlgorithm):
    class__ = "KernelPCA"
    module__ = "sklearn.decomposition"

