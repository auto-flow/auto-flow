from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__=["FastICA"]

class FastICA(AutoPLFeatureEngineerAlgorithm):
    class__ = "FastICA"
    module__ = "sklearn.decomposition"
