import numpy as np
from scipy import sparse
from sklearn.base import ClassifierMixin

from autoflow.ensemble.stack.base import StackEstimator

__all__ = ["StackClassifier"]


class StackClassifier(StackEstimator, ClassifierMixin):
    mainTask = "classification"

    def __init__(
            self,
            meta_learner=None,
            use_features_in_secondary=False,
            drop_last_proba=False,
            use_probas=True
    ):
        super(StackClassifier, self).__init__(meta_learner, use_features_in_secondary)
        self.use_probas = use_probas
        self.drop_last_proba = drop_last_proba

    def predict_meta_features(self, X, is_train):

        per_model_preds = []

        for i, models in enumerate(self.estimators_list):
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

    def predict_proba(self, X):
        return self._do_predict(X, self.meta_learner.predict_proba)

    def decision_function(self, X):
        return self._do_predict(X, self.meta_learner.decision_function)
