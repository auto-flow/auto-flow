from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__=["TruncatedSVD"]

class TruncatedSVD(AutoPLPreprocessingAlgorithm):
    class__ = "TruncatedSVD"
    module__ = "sklearn.decomposition"