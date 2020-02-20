from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm


class MinMaxScaler( AutoPLPreprocessingAlgorithm):
    class__ = "MinMaxScaler"
    module__ = "sklearn.preprocessing"


