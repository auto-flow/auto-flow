from autoflow.manager.data_container.dataframe import DataFrameContainer
from autoflow.pipeline.components.classification_base import AutoFlowClassificationAlgorithm
from autoflow.utils.data import softmax

__all__ = ["SGD"]


class SGD(
    AutoFlowClassificationAlgorithm
):
    module__ = "sklearn.linear_model.stochastic_gradient"
    class__ = "SGDClassifier"

    def predict_proba(self, X: DataFrameContainer):
        if self.hyperparams["loss"] in ["log", "modified_huber"]:
            return super(SGD, self).predict_proba(X)
        else:
            df = self.estimator.decision_function(X.data)
            return softmax(df)
