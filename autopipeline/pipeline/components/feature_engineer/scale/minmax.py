from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__=["MinMaxScaler"]

class MinMaxScaler( AutoPLPreprocessingAlgorithm):
    class__ = "MinMaxScaler"
    module__ = "sklearn.preprocessing"


