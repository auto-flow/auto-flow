from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm

__all__=["TruncatedSVD"]

class TruncatedSVD(AutoPLPreprocessingAlgorithm):
    class__ = "TruncatedSVD"
    module__ = "sklearn.decomposition"