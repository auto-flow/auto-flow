from sklearn.preprocessing import StandardScaler

from autopipeline.pipeline.components.base import AutoPLRegressionAlgorithm


class SGD(
    AutoPLRegressionAlgorithm,
):
    module__ = "sklearn.linear_model.stochastic_gradient"
    class__ = "SGDRegressor"

    def after_process_fit_y(self, y):
        self.scaler = StandardScaler(copy=True)
        return self.scaler.fit(y.reshape((-1, 1))).ravel()

    def after_process_pred_y(self, y):
        return self.scaler.inverse_transform(y)
