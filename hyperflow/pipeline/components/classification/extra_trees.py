from hyperflow.pipeline.components.classification_base import HyperFlowClassificationAlgorithm

__all__=["ExtraTreesClassifier"]


class ExtraTreesClassifier(
    HyperFlowClassificationAlgorithm,
):
    class__ = "ExtraTreesClassifier"
    module__ = "sklearn.ensemble"
    tree_model = True
