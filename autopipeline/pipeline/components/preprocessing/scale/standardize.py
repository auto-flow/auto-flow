from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__=["StandardScaler"]

class StandardScaler(AutoPLFeatureEngineerAlgorithm):
    class__ = "StandardScaler"
    module__ = "sklearn.preprocessing"
