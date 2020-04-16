import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder

from autoflow.utils.data import to_array

__all__ = ["LabelEncoder"]


class LabelEncoder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        X = to_array(X)
        encoders = []
        for i in range(X.shape[1]):
            cur = X[:, i]
            encoder = SklearnLabelEncoder().fit(cur)  # [cur != -999]
            encoders.append(encoder)
        self.encoders = encoders
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            columns = X.columns
            index = X.index
        else:
            columns = [str(i) for i in range(X.shape[1])]
            index = range(X.shape[0])
        X = to_array(X)
        arrs = []
        assert X.shape[1] == len(self.encoders)
        for i in range(X.shape[1]):
            cur = X[:, i]
            # arr = np.zeros_like(cur)
            encoder = self.encoders[i]
            arr = encoder.transform(cur)
            # arr[cur == -999] = -999
            arrs.append(arr)
        return pd.DataFrame(np.vstack(arrs).T, columns=columns, index=index)
