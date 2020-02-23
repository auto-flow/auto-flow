from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm

__all__=["MinMaxScaler"]

class MinMaxScaler( AutoPLPreprocessingAlgorithm):
    class__ = "MinMaxScaler"
    module__ = "sklearn.preprocessing"


