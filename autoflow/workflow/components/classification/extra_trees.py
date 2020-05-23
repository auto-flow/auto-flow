from autoflow.workflow.components.base import AutoFlowIterComponent
from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm

__all__=["ExtraTreesClassifier"]


class ExtraTreesClassifier(
    AutoFlowIterComponent, AutoFlowClassificationAlgorithm,
):
    class__ = "ExtraTreesClassifier"
    module__ = "sklearn.ensemble"
    tree_model = True
