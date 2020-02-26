from autopipeline.pipeline.components.classification_base import AutoPLClassificationAlgorithm

__all__=["ExtraTreesClassifier"]


class ExtraTreesClassifier(
    AutoPLClassificationAlgorithm,
):
    class__ = "ExtraTreesClassifier"
    module__ = "sklearn.ensemble"
