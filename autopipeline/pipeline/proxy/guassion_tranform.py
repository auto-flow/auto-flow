import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import TransformerMixin, BaseEstimator


class GuassionTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self._type = "DataFrame"

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            self._type = "ndarray"
        skew_array = stats.skew(X)
        minimum_array = map(lambda x: min(X[x]), X.columns)
        std_array = np.std(X)
        self.transformer = dict(zip(X.columns.values, zip(std_array, minimum_array, skew_array)))
        for ii, column in enumerate(X.columns.values):
            one = X[column]
            std, skew, minimum = self.transformer.get(column)
            if skew > 2 * std:
                if skew > 3 * std:
                    if minimum < 0:
                        one = np.log10(one + np.array([2 * std + minimum + 1] * X.shape[0]))
                    else:
                        one = np.log10(one)
                else:
                    if minimum < 0:
                        one = np.sqrt(one + np.array([2 * std + minimum + 1] * X.shape[0]))
                    else:
                        one = np.sqrt(one)
                if abs(stats.skew(one)) < abs(skew):
                    X[column] = one
        return X.values


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target
    GuassionTransformer().fit(X,y)