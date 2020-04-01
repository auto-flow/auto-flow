from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm

__all__=["TruncatedSVD"]

class TruncatedSVD(HyperFlowFeatureEngineerAlgorithm):
    class__ = "TruncatedSVD"
    module__ = "sklearn.decomposition"