from sklearn.base import TransformerMixin, BaseEstimator

__all__ = ["NoPreprocessing"]


class NoPreprocessing(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        self.preprocessor = 0
        return self

    def transform(self, X):
        return X
