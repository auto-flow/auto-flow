from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__=["TruncatedSVD"]

class TruncatedSVD(AutoPLFeatureEngineerAlgorithm):
    class__ = "TruncatedSVD"
    module__ = "sklearn.decomposition"