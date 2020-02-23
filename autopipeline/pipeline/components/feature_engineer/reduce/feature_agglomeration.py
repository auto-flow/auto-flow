from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm


class FeatureAgglomeration(AutoPLPreprocessingAlgorithm):
    module__ = "sklearn.cluster"
    class__ = "FeatureAgglomeration"