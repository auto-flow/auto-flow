import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder
import pandas as pd

__all__ = ["LabelEncoder"]


class LabelEncoder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        if isinstance(X,pd.DataFrame):
            X=X.values
        encoders = []
        for i in range(X.shape[1]):
            encoder = SklearnLabelEncoder().fit(X[:, i].ravel())
            encoders.append(encoder)
        self.encoders = encoders
        return self

    def transform(self, X, y=None):
        if isinstance(X,pd.DataFrame):
            X=X.values
        arrs = []
        assert X.shape[1] == len(self.encoders)
        for i in range(X.shape[1]):
            encoder = self.encoders[i]
            arr = encoder.transform(X[:, i].ravel())
            arrs.append(arr)
        return np.vstack(arrs).T

    # def fit_transform(self, X, y=None, **fit_params):
    #     self.fit(X,y)
    #     return self.transform(X,y)

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv("/home/tqc/PycharmProjects/auto-pipeline/examples/classification/train_classification.csv")
    encoded=LabelEncoder().fit_transform(df[["Sex","Cabin"]].fillna("nan"))
    print(encoded)
