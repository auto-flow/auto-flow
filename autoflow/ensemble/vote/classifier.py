from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class VoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,models):
        self.models = models

    def predict(self,X):
        return np.argmax(self.predict_proba(X),axis=1)

    def predict_proba(self,X):
        probas = [model.predict_proba(X) for model in self.models]
        probas_arr = np.array(probas)
        proba = np.average(probas_arr, axis=0)
        return proba


