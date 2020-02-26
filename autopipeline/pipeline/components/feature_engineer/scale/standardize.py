from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__=["StandardScaler"]

class StandardScaler(AutoPLPreprocessingAlgorithm):
    class__ = "StandardScaler"
    module__ = "sklearn.preprocessing"
