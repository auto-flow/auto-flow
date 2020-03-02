from typing import List

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin


class StackingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
            self,
            meta_learner,
            models_list: List,
            prediction_list: List,
            drop_last_proba=False,
            use_features_in_secondary=False,
            use_probas=True
    ):

        self.use_probas = use_probas
        self.prediction_list = prediction_list
        self.models_list = models_list
        self.meta_learner = meta_learner
        self.use_features_in_secondary = use_features_in_secondary
        self.drop_last_proba = drop_last_proba

    def fit(self, X, y):
        meta_features = self.predict_meta_features(X, True)
        self.meta_learner.fit(meta_features, y)

    def predict_meta_features(self, X, is_train):

        per_model_preds = []

        for i, models in enumerate(self.models_list):
            if is_train:
                proba = self.prediction_list[i]
            else:
                probas = [model.predict_proba(X) for model in models]
                probas_arr = np.array(probas)
                proba = np.average(probas_arr, axis=0)
            if not self.use_probas:
                prediction = np.argmax(proba, axis=1)
            else:
                if self.drop_last_proba:
                    prediction = proba[:, :-1]
                else:
                    prediction = proba

            per_model_preds.append(prediction)

        meta_features = np.hstack(per_model_preds)
        if not self.use_features_in_secondary:
            return (meta_features)
        elif sparse.issparse(X):
            return (sparse.hstack((X, meta_features)))
        else:
            return (np.hstack((X, meta_features)))

    def _do_predict(self, X, predict_fn):
        meta_features = self.predict_meta_features(X, False)
        return predict_fn(meta_features)

    def predict(self, X):
        return self._do_predict(X, self.meta_learner.predict)

    def predict_proba(self, X):
        return self._do_predict(X, self.meta_learner.predict_proba)

    def decision_function(self, X):
        return self._do_predict(X, self.meta_learner.decision_function)