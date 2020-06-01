from autoflow.data_container import DataFrameContainer
from autoflow.utils.data import softmax
from autoflow.workflow.components.base import AutoFlowIterComponent
from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm

__all__ = ["SGDClassifier"]


class SGDClassifier(AutoFlowIterComponent, AutoFlowClassificationAlgorithm):
    module__ = "sklearn.linear_model.stochastic_gradient"
    class__ = "SGDClassifier"
    iterations_name = "max_iter"

    def predict_proba(self, X: DataFrameContainer):
        if self.hyperparams["loss"] in ["log", "modified_huber"]:
            return super(SGDClassifier, self).predict_proba(X)
        else:
            df = self.component.decision_function(X.data)
            return softmax(df)
