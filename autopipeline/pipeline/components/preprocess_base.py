from autopipeline.pipeline.components.base import AutoPLComponent


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

    def _pred_or_trans(self, X_train, X_valid=None, X_test=None, is_train=False):
        X_train = self.estimator.transform(X_train)
        X_valid = self.estimator.transform(X_valid)
        X_test = self.estimator.transform(X_test)
