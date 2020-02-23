from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm

__all__=["RobustScalerComponent"]

class RobustScalerComponent(AutoPLPreprocessingAlgorithm):
    class__ = "RobustScaler"
    module__ = "sklearn.preprocessing"
