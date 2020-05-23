from typing import Dict

from autoflow.workflow.components.base import AutoFlowIterComponent
from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm

__all__ = ["RandomForestClassifier"]


class RandomForestClassifier(AutoFlowIterComponent, AutoFlowClassificationAlgorithm):
    class__ = "RandomForestClassifier"
    module__ = "sklearn.ensemble"


