from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm


class RobustScalerComponent(AutoPLPreprocessingAlgorithm):
    class__ = "RobustScaler"
    module__ = "sklearn.preprocessing"
