import numpy as np

from autopipeline.pipeline.components.base import AutoPLComponent
from autopipeline.pipeline.dataframe import GenericDataFrame
from autopipeline.utils.data import densify


class AutoPLPreprocessingAlgorithm(AutoPLComponent):

    # def transform(self, X):
    #     X=densify(X)
    #     if not self.estimator or (not hasattr(self.estimator, "transform")):
    #         raise NotImplementedError()
    #     X=self.before_pred_X(X)
    #     return self.after_pred_X(self.estimator.transform(X))

    def fit_transform(self, X_train=None, y_train=None, X_valid=None, y_valid=None, X_test=None, y_test=None,
                      is_train=True):
        self.fit(X_train, y_train, X_valid, y_valid, X_test, y_test)
        return self.transform(X_train, X_valid, X_test, is_train)

    def transform(self, X_train=None, X_valid=None, X_test=None, is_train=False):
        return self.pred_or_trans(X_train, X_valid, X_test, is_train)

    def _transform_proc(self, X):
        if X is None:
            return None
        else:
            return self.estimator.transform(X)

    def _transform(self, X_: np.ndarray, X: GenericDataFrame):
        if X_ is None:
            return None
        X_ = self._transform_proc(X_)
        X_ = densify(X_)  # todo: 改为判断的形式？
        X_ = self.before_trans_X(X_)
        return X.replace_feat_grp(self.in_feat_grp, X_, self.out_feat_grp)

    def _pred_or_trans(self, X_train_, X_valid_=None, X_test_=None, X_train=None, X_valid=None, X_test=None,
                       is_train=False):
        X_train = self._transform(X_train_, X_train)
        X_valid = self._transform(X_valid_, X_valid)
        X_test = self._transform(X_test_, X_test)
        return {
            "X_train": X_train,
            "X_valid": X_valid,
            "X_test": X_test,
        }

    def before_trans_X(self, X):
        return X
