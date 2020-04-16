import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

__all__=["DropAll"]

class DropAll(TransformerMixin, BaseEstimator):


    def fit(self,X=None,y=None):
        return self

    def transform(self, X, y=None):
        return np.zeros([X.shape[0], 0])


    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X,y)
        return self.transform(X,y)