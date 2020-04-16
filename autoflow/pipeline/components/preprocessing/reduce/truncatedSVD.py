from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__=["TruncatedSVD"]

class TruncatedSVD(AutoFlowFeatureEngineerAlgorithm):
    class__ = "TruncatedSVD"
    module__ = "sklearn.decomposition"