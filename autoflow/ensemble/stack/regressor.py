import numpy as np
from sklearn.base import RegressorMixin

from autoflow.ensemble.stack.base import StackEstimator

__all__=["StackRegressor"]

class StackRegressor(StackEstimator, RegressorMixin):
    mainTask = "regression"

    def predict_meta_features(self, X, is_train):

        per_model_preds = []

        for i, models in enumerate(self.estimators_list):
            if is_train:
                prediction = self.prediction_list[i]
            else:
                probas = [model.predict(X) for model in models]
                probas_arr = np.array(probas)
                prediction = np.average(probas_arr, axis=0)
            per_model_preds.append(prediction)

        meta_features = np.vstack(per_model_preds).T # todo: 和classifier 对比， 更好的方法
        return (meta_features)
