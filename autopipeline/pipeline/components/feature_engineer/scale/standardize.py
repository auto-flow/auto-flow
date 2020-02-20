from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm


class StandardScaler(AutoPLPreprocessingAlgorithm):
    class__ = "StandardScaler"
    module__ = "sklearn.preprocessing"
