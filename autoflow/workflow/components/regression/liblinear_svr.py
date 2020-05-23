from autoflow.manager.data_container.ndarray import NdArrayContainer
from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm
from sklearn.preprocessing import StandardScaler

class LibLinear_SVR(AutoFlowRegressionAlgorithm):
    class__ = "LinearSVR"
    module__ = "sklearn.svm"

    def before_fit_y(self, y:NdArrayContainer):
        if y is None:
            return None
        y = deepcopy(y.data)
        self.scaler = StandardScaler(copy=True)
        y = y.ravel().reshape([-1, 1])
        self.scaler.fit(y)
        return self.scaler.transform(y)

    def after_pred_y(self, y):
        return self.scaler.inverse_transform(y)
