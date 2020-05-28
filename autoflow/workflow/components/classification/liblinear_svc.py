from autoflow.manager.data_container.dataframe import DataFrameContainer
from autoflow.utils.data import softmax
from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm

__all__ = ["LibLinear_SVC"]


class LibLinear_SVC(AutoFlowClassificationAlgorithm):
    class__ = "LinearSVC"
    module__ = "sklearn.svm"
    OVR__ = True

    def predict_proba(self, X: DataFrameContainer):
        df = self.component.decision_function(X.data)
        return softmax(df)