from autoflow.pipeline.components.classification_base import AutoFlowClassificationAlgorithm

__all__=["ExtraTreesClassifier"]


class ExtraTreesClassifier(
    AutoFlowClassificationAlgorithm,
):
    class__ = "ExtraTreesClassifier"
    module__ = "sklearn.ensemble"
    tree_model = True
