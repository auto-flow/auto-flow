from autoflow.workflow.components.base import AutoFlowIterComponent
from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm

__all__=["LogisticRegression"]

class LogisticRegression(AutoFlowIterComponent, AutoFlowClassificationAlgorithm):
    class__ = "LogisticRegression"
    module__ = "sklearn.linear_model"
    iterations_name = "max_iter"
