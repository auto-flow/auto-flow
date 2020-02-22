from autopipeline.pipeline.components.base import AutoPLClassificationAlgorithm


class ExtraTreesClassifier(
    AutoPLClassificationAlgorithm,
):
    class__ = "ExtraTreesClassifier"
    module__ = "sklearn.ensemble"
