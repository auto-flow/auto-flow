import  pandas as pd

class SklearnSelectMixin():
    def _transform_proc(self, X):
        if X is None:
            return None
        else:
            trans = self.estimator.transform(X)
            mask = self.estimator.get_support()
            columns=X.columns[mask]
            return pd.DataFrame(trans,columns=columns)