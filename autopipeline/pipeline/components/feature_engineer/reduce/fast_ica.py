from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm


class FastICA(AutoPLPreprocessingAlgorithm):
    class__ = "FastICA"
    module__ = "sklearn.decomposition"
