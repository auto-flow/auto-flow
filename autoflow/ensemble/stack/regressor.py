import numpy as np
from sklearn.base import RegressorMixin

from autoflow.ensemble.stack.base import StackEstimator

__all__=["StackRegressor"]

class StackRegressor(StackEstimator, RegressorMixin):
    ml_task = "regression"

    def predict_meta_features(self, X, is_train):

        per_model_preds = []

        for i, models in enumerate(self.estimators_list):
            if is_train:
                proba = self.prediction_list[i]
            else:
                probas = [model.predict(X) for model in models]
                probas_arr = np.array(probas)
                proba = np.average(probas_arr, axis=0)
            prediction = np.argmax(proba, axis=1)
            per_model_preds.append(prediction)

        meta_features = np.hstack(per_model_preds)
        return (meta_features)
