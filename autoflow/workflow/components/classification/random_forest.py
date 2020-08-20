from typing import Dict

from autoflow.workflow.components.iter_algo import AutoFlowIterComponent
from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm

__all__ = ["RandomForestClassifier"]


class RandomForestClassifier(AutoFlowIterComponent, AutoFlowClassificationAlgorithm):
    class__ = "RandomForestClassifier"
    module__ = "sklearn.ensemble"


