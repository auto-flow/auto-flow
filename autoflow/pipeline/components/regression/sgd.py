from sklearn.preprocessing import StandardScaler

from autoflow.manager.data_container.ndarray import NdArrayContainer
from autoflow.pipeline.components.regression_base import AutoFlowRegressionAlgorithm


class SGD(
    AutoFlowRegressionAlgorithm,
):
    module__ = "sklearn.linear_model.stochastic_gradient"
    class__ = "SGDRegressor"

    def before_fit_y(self, y: NdArrayContainer):
        if y is None:
            return None
        y = y.data
        self.scaler = StandardScaler(copy=True)
        return self.scaler.fit(y.reshape((-1, 1))).ravel()

    def after_pred_y(self, y):
        return self.scaler.inverse_transform(y)
