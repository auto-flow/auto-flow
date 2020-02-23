from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm


class TruncatedSVD(AutoPLPreprocessingAlgorithm):
    class__ = "TruncatedSVD"
    module__ = "sklearn.decomposition"